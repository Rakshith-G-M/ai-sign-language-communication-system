"""
src/inference/text_builder.py

Text construction module for a real-time sign language recognition system.
Converts stable per-frame letter predictions into words and sentences.

Spell correction:
    SymSpell (max_edit_distance=2) is applied once per completed word inside
    _commit_word().  It is never called per-frame or per-letter so it adds
    negligible latency to the real-time loop.
"""

import asyncio
import os
import tempfile
import threading
import time
import pkg_resources
import edge_tts
import pygame
from symspellpy import SymSpell, Verbosity


class TextBuilder:
    """
    Converts a stream of stable letter predictions into words and sentences.

    Transitions:
        Letter received   → append to current_word (if changed)
        No hand > 1 s     → commit current_word to sentence (space)
        No hand > 3 s     → finalise and print sentence, then reset all state

    Spell correction:
        Applied once in _commit_word() before the word enters the sentence.
        Examples:  HEILO → HELLO   WOERLD → WORLD
    """

    # ------------------------------------------------------------------ #
    #  Tuneable thresholds                                                 #
    # ------------------------------------------------------------------ #
    SPACE_TIMEOUT: float = 1.0      # seconds of absent hand → word break
    SENTENCE_TIMEOUT: float = 3.0   # seconds of absent hand → sentence end

    def __init__(self) -> None:
        self.current_word: str = ""
        self.sentence: str = ""

        self._last_letter: str | None = None   # duplicate-guard
        self._last_hand_time: float = time.time()
        self._space_committed: bool = False    # prevent repeated space commits

        # ── Letter timing (noise filter + repeat support) ──────────────────
        # _letter_start_time : when the current stable letter was first seen.
        # _last_added_time   : when a letter was last appended to current_word.
        # MIN_LETTER_DURATION: how long a letter must be held before it is
        #                      accepted — filters fleeting noise frames.
        # REPEAT_COOLDOWN    : minimum gap before the same letter may be
        #                      appended again (HEL → L → HELL, blocks LLLL).
        self._letter_start_time: float = 0.0
        self._last_added_time: float = 0.0
        self.MIN_LETTER_DURATION: float = 0.6   # seconds held before accepted
        self.REPEAT_COOLDOWN: float = 1.2        # seconds before same letter repeats

        # ── SymSpell initialisation ────────────────────────────────────────
        # Loaded once at construction; lookup() is O(1) at runtime so it is
        # safe to call inside the real-time loop indirectly via _commit_word().
        self._sym_spell = SymSpell(max_dictionary_edit_distance=2)
        _dict_path = pkg_resources.resource_filename(
            "symspellpy",
            "frequency_dictionary_en_82_765.txt",
        )
        self._sym_spell.load_dictionary(_dict_path, term_index=0, count_index=1)

        # ── pygame mixer (Edge TTS playback) ──────────────────────────────
        # Initialised once here; play() is non-blocking so the real-time
        # prediction loop continues uninterrupted while audio plays.
        pygame.mixer.init()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        stable_letter: str | None,
        hand_detected: bool,
        timestamp: float,
    ) -> tuple[str, str]:
        """
        Process one prediction tick.

        Parameters
        ----------
        stable_letter : str | None
            The currently stable letter, or None if no confident prediction.
        hand_detected : bool
            Whether a hand is visible in the current frame.
        timestamp : float
            Current wall-clock time (``time.time()``).

        Returns
        -------
        current_word : str
            The word being built right now (uncorrected — raw signed letters).
        sentence : str
            The sentence accumulated so far (spell-corrected words).
        """
        if hand_detected:
            self._on_hand_present(stable_letter, timestamp)
        else:
            self._on_hand_absent(timestamp)

        return self.current_word, self.sentence

    def reset(self) -> None:
        """Hard-reset all state (useful for testing or manual restarts)."""
        self.current_word = ""
        self.sentence = ""
        self._last_letter = None
        self._last_hand_time = time.time()
        self._space_committed = False

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _on_hand_present(self, stable_letter: str | None, timestamp: float) -> None:
        """
        Called every tick while a hand is detected.

        Letter acceptance rules
        ───────────────────────
        New letter  → start the hold timer; do NOT append yet.
        Same letter held ≥ MIN_LETTER_DURATION → eligible to append.
            First occurrence : append immediately (word empty or last char
                               differs from incoming letter).
            Repeat           : append only after REPEAT_COOLDOWN has elapsed
                               since the last append, enabling intentional
                               doubled letters (HEL → hold-L → HELL) while
                               blocking unintentional spam (LLLL).
        After appending, reset _letter_start_time so the next repeat requires
        another full MIN_LETTER_DURATION hold.
        """
        self._last_hand_time = timestamp
        self._space_committed = False   # hand is back; re-arm space logic

        if stable_letter is None:
            return

        if stable_letter != self._last_letter:
            # New letter detected → start hold timer, do not append yet
            self._letter_start_time = timestamp
            self._last_letter = stable_letter
        else:
            # Same letter held → enforce minimum duration before accepting
            if (timestamp - self._letter_start_time) > self.MIN_LETTER_DURATION:
                # Append if: word is empty, last char differs, or cooldown
                # has elapsed (intentional repeat such as HEL → L → HELL).
                if (
                    not self.current_word
                    or self.current_word[-1] != stable_letter
                    or (timestamp - self._last_added_time) > self.REPEAT_COOLDOWN
                ):
                    self.current_word += stable_letter
                    self._last_added_time = timestamp
                    # Reset so the next repeat needs another full cooldown
                    self._letter_start_time = timestamp

    def _on_hand_absent(self, timestamp: float) -> None:
        """Called every tick while no hand is detected."""
        elapsed = timestamp - self._last_hand_time

        if elapsed > self.SENTENCE_TIMEOUT:
            self._finalise_sentence()
        elif elapsed > self.SPACE_TIMEOUT:
            self._commit_word()

    def _correct_word(self, word: str) -> str:
        """
        Return the closest dictionary match for *word* using SymSpell.

        Correction is bounded to max_edit_distance=2 (set at construction).
        If no suggestion is found the original word is returned unchanged,
        preserving any intentional abbreviations or proper nouns.

        Called exclusively from _commit_word() and _finalise_sentence()
        — never per-frame or per-letter.
        """
        suggestions = self._sym_spell.lookup(
            word.lower(),           # dictionary is lowercase; normalise input
            Verbosity.CLOSEST,
            max_edit_distance=2,
        )
        if suggestions:
            # Mirror the original word's casing style (all-caps for ASL output)
            corrected = suggestions[0].term
            return corrected.upper() if word.isupper() else corrected
        return word

    def _commit_word(self) -> None:
        """
        Flush current_word into sentence as a space-separated token.

        The word is spell-checked via SymSpell before being appended so that
        minor signing errors (e.g. HEILO → HELLO) are silently corrected at
        the word boundary without touching the real-time prediction pipeline.
        """
        if self._space_committed:
            return                              # already committed for this gap
        if not self.current_word:
            return                              # nothing to commit

        corrected_word = self._correct_word(self.current_word)  # ← spell check
        self.sentence += corrected_word + " "
        self.current_word = ""
        self._last_letter = None
        self._space_committed = True

    async def _speak_async(self, text: str) -> None:
        """
        Synthesise *text* with Edge TTS and play it via pygame (non-blocking).

        Edge TTS streams a high-quality neural voice from Microsoft's cloud
        and saves it to a temp .mp3.  pygame.mixer.music.play() returns
        immediately so the real-time prediction loop is never stalled.
        A daemon thread polls get_busy() and removes the temp file once
        playback ends so temp files do not accumulate.

        Voice: en-US-AriaNeural (natural female, US English).
        """
        communicate = edge_tts.Communicate(
            text=text.lower(),          # consistent pronunciation for ALL-CAPS
            voice="en-US-AriaNeural",
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        await communicate.save(temp_path)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()       # non-blocking

        def _cleanup(path: str) -> None:
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            try:
                os.remove(path)
            except OSError:
                pass
        threading.Thread(target=_cleanup, args=(temp_path,), daemon=True).start()

    def _finalise_sentence(self) -> None:
        """Commit any trailing word, print the finished sentence, then reset."""
        # Flush any word that hasn't been committed yet (spell-check included)
        if self.current_word:
            corrected_word = self._correct_word(self.current_word)  # ← spell check
            self.sentence += corrected_word + " "
            self.current_word = ""
            self._last_letter = None

        if self.sentence.strip():
            text = self.sentence.strip()
            print(f"[Sentence] {text}")
            try:
                asyncio.run(self._speak_async(text))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Edge TTS speech failed: %s", exc
                )

        # Full reset so the next sentence starts clean
        self.sentence = ""
        self._space_committed = False