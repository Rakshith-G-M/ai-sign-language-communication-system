"""
text_builder.py

Text construction module for a real-time sign language recognition system.
Converts stable per-frame letter predictions into words and sentences.

Pipeline
────────
    Stabilized letter   →   accumulate in current_word (duplicate suppressed)
    No hand > 2 s       →   commit current_word to sentence
    No hand > 5 s       →   finalise and reset sentence

Spell correction:
    SymSpell (max_edit_distance=2) is applied once per completed word inside
    _commit_word().  It is never called per-frame or per-letter so it adds
    negligible latency to the real-time loop.
"""

import time
import logging
from enum import Enum
from symspellpy import SymSpell, Verbosity

log = logging.getLogger(__name__)


class GestureState(Enum):
    IDLE = 0
    DETECTING = 1
    LOCKED = 2


class TextBuilder:
    """
    Converts a stream of stable letter predictions into words and sentences.

    Transitions:
        Letter received   → append to current_word (if changed and held long enough)
        No hand > 2 s     → commit current_word to sentence (space)
        No hand > 5 s     → finalise and reset sentence

    Spell correction:
        Applied once in _commit_word() before the word enters the sentence.
        Examples:  HEILO → HELLO   WOERLD → WORLD
    """

    # ------------------------------------------------------------------ #
    #  Tuneable thresholds                                                 #
    # ------------------------------------------------------------------ #
    SPACE_TIMEOUT: float = 2.0      # seconds of absent hand → word break
    SENTENCE_TIMEOUT: float = 5.0   # seconds of absent hand → reset sentence
    GESTURE_RESET_TIMEOUT: float = 0.8  # seconds of absent hand → wipe gesture lock

    def __init__(self) -> None:
        self.current_word: str = ""
        self.sentence: str = ""

        self._last_letter: str | None = None   # duplicate-guard
        self._last_hand_time: float = time.time()
        self._space_committed: bool = False    # prevent repeated space commits
        self._last_finalized_sentence: str | None = None  # for TTS signaling
        self._last_spoken_sentence: str | None = None   # deduction memory

        # ── State Machine ──────────────────────────────────────────────────
        self._state: GestureState = GestureState.IDLE
        self._letter_start_time: float = 0.0
        self._last_added_time: float = 0.0

        # ── Letter timing (noise filter) ──────────────────────────────────
        self.MIN_LETTER_DURATION: float = 0.6   # seconds held before LOCKED

        # ── SymSpell initialisation ────────────────────────────────────────
        # Loaded once at construction; lookup() is O(1) at runtime so it is
        # safe to call inside the real-time loop indirectly via _commit_word().
        import os
        import symspellpy
        self._sym_spell = SymSpell(max_dictionary_edit_distance=2)
        symspellpy_dir = os.path.dirname(symspellpy.__file__)
        _dict_path = os.path.join(symspellpy_dir, "frequency_dictionary_en_82_765.txt")
        self._sym_spell.load_dictionary(_dict_path, term_index=0, count_index=1)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        stable_letter: str | None,
        hand_detected: bool,
        timestamp: float,
    ) -> tuple[str, str, list[str]]:
        """
        Process one prediction tick from the static alphabet pipeline.

        Args:
            stable_letter: Stabilised ASL letter from the prediction layer,
                           or None when no confident letter is detected.
            hand_detected: True when MediaPipe found a hand in the frame.
            timestamp:     Current wall-clock time (time.time()).

        Returns:
            (current_word, sentence, suggestions)
        """
        if hand_detected:
            self._on_hand_present(stable_letter, timestamp)
        else:
            self._on_hand_absent(timestamp)

        suggestions = self.get_word_suggestions()
        return self.current_word, self.sentence, suggestions

    def reset(self) -> None:
        """Hard-reset all state (useful for testing or manual restarts)."""
        self.current_word = ""
        self.sentence = ""
        self._last_letter = None
        self._last_hand_time = time.time()
        self._space_committed = False
        self._state = GestureState.IDLE
        self._last_finalized_sentence = None
        self._last_spoken_sentence = None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _on_hand_present(self, stable_letter: str | None, timestamp: float) -> None:
        """
        Called every tick while a hand is detected.

        Letter acceptance rules
        ───────────────────────
        A new letter is only appended to current_word once it has been held
        continuously for MIN_LETTER_DURATION seconds (DETECTING → LOCKED).
        While the same letter is held in LOCKED state, it is not re-emitted,
        preventing duplicate characters from a sustained sign hold.
        A different letter immediately resets the state to DETECTING.
        """
        self._last_hand_time = timestamp
        self._space_committed = False

        if stable_letter is None:
            return

        # New letter detected → restart the hold timer
        if stable_letter != self._last_letter:
            self._state = GestureState.DETECTING
            self._letter_start_time = timestamp
            self._last_letter = stable_letter
            return

        # DETECTING → LOCKED: letter has been held long enough
        if self._state == GestureState.DETECTING:
            if (timestamp - self._letter_start_time) > self.MIN_LETTER_DURATION:
                self._state = GestureState.LOCKED
                self.current_word += stable_letter
                self._last_added_time = timestamp
                # Stays LOCKED; no re-emission until state resets to DETECTING

    def _on_hand_absent(self, timestamp: float) -> None:
        """State transitions when hand is absent."""
        elapsed = timestamp - self._last_hand_time

        # Hysteresis: only move back to IDLE after a significant absence
        if elapsed > self.GESTURE_RESET_TIMEOUT:
            self._state = GestureState.IDLE
            self._last_letter = None

        if elapsed > self.SENTENCE_TIMEOUT:
            self._finalise_sentence()
        elif elapsed > self.SPACE_TIMEOUT:
            self._commit_word()

    def get_word_suggestions(self) -> list[str]:
        """
        Prefix matching with frequency ranking using SymSpell's dictionary.
        Returns up to 3 capitalised suggestions for the current partial word.
        """
        if not self.current_word:
            return []

        prefix = self.current_word.lower()
        matches = []

        for word, count in self._sym_spell.words.items():
            if word.startswith(prefix):
                matches.append((word, count))
                if len(matches) > 100:   # cap search for speed
                    break

        matches.sort(key=lambda x: (x[0].lower() == prefix, x[1]), reverse=True)
        return [m[0].upper() for m in matches[:3]]

    def _correct_word(self, word: str) -> str:
        """
        Return the closest dictionary match for *word* using SymSpell.

        Bounded to max_edit_distance=2.  Returns the original word unchanged
        when no suggestion is found, preserving abbreviations and proper nouns.
        Called exclusively from _commit_word() — never per-frame or per-letter.
        """
        suggestions = self._sym_spell.lookup(
            word.lower(),
            Verbosity.CLOSEST,
            max_edit_distance=2,
        )
        if suggestions:
            corrected = suggestions[0].term
            return corrected.upper() if word.isupper() else corrected
        return word

    def _commit_word(self) -> None:
        """Flush current_word into the sentence with spell correction."""
        if self._space_committed or not self.current_word:
            return

        raw_word = self.current_word

        # Exact match: keep as-is to avoid "HI" → "HIS"
        if raw_word.lower() in self._sym_spell.words:
            chosen_word = raw_word
        else:
            chosen_word = self._correct_word(raw_word)

        self.sentence += chosen_word + " "
        self.current_word = ""
        self._last_letter = None
        self._space_committed = True

    def _finalise_sentence(self) -> None:
        """Finalise sentence with memory protection to prevent repeated triggers."""
        if self.current_word:
            self._commit_word()

        final_text = self.sentence.strip()
        if not final_text:
            return

        if final_text != self._last_spoken_sentence:
            log.info("[Sentence] %s", final_text)
            self._last_finalized_sentence = final_text
            self._last_spoken_sentence = final_text

        self._space_committed = False

    def pop_final_sentence(self) -> str | None:
        """
        Returns the completed sentence once for TTS, then clears the marker.
        Does NOT clear the full sentence history, just the 'new' marker.
        """
        if self._last_finalized_sentence:
            text = self._last_finalized_sentence
            self._last_finalized_sentence = None
            return text
        return None