"""
src/inference/text_builder.py

Text construction module for a real-time sign language recognition system.
Converts stable per-frame letter predictions into words and sentences.

Spell correction:
    SymSpell (max_edit_distance=2) is applied once per completed word inside
    _commit_word().  It is never called per-frame or per-letter so it adds
    negligible latency to the real-time loop.
"""

import time
import pkg_resources
from enum import Enum
from symspellpy import SymSpell, Verbosity


class GestureState(Enum):
    IDLE = 0
    DETECTING = 1
    LOCKED = 2


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
    SPACE_TIMEOUT: float = 2.0      # seconds of absent hand → word break
    SENTENCE_TIMEOUT: float = 5.0   # seconds of absent hand → reset sentence

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
        self._sym_spell = SymSpell(max_dictionary_edit_distance=2)
        _dict_path = pkg_resources.resource_filename(
            "symspellpy",
            "frequency_dictionary_en_82_765.txt",
        )
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
        Process one prediction tick.
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

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _on_hand_present(self, stable_letter: str | None, timestamp: float) -> None:
        """
        Called every tick while a hand is detected.

        Letter acceptance rules
        ───────────────────────
        State transitions when a hand is detected.
        """
        self._last_hand_time = timestamp
        self._space_committed = False

        if stable_letter is None:
            return

        # SIGNIFICANT CHANGE Detection → Release Lock
        if stable_letter != self._last_letter:
            self._state = GestureState.DETECTING
            self._letter_start_time = timestamp
            self._last_letter = stable_letter
            return

        # STATE: DETECTING → Potential transition to LOCKED
        if self._state == GestureState.DETECTING:
            if (timestamp - self._letter_start_time) > self.MIN_LETTER_DURATION:
                self._state = GestureState.LOCKED
                self.current_word += stable_letter
                self._last_added_time = timestamp
                # Note: No re-emission until state changes back to DETECTING or IDLE

    GESTURE_RESET_TIMEOUT: float = 0.8 # seconds of absent hand → wipe gesture lock

    def _on_hand_absent(self, timestamp: float) -> None:
        """State transitions when hand is absent."""
        elapsed = timestamp - self._last_hand_time

        # HYSTERESIS: Only move back to IDLE if hand is gone for a significant time
        if elapsed > self.GESTURE_RESET_TIMEOUT:
            self._state = GestureState.IDLE
            self._last_letter = None
        
        if elapsed > self.SENTENCE_TIMEOUT:
            self._finalise_sentence()
        elif elapsed > self.SPACE_TIMEOUT:
            self._commit_word()

    def get_word_suggestions(self) -> list[str]:
        """
        Smarter language layer: Prefix matching + Frequency ranking.
        Using SymSpell's internal dictionary as the frequency source.
        """
        if not self.current_word:
            return []
        
        prefix = self.current_word.lower()
        matches = []
        
        # Iterating over the dictionary (optimized for performance in real-time)
        # We only look for the first 10,000 matches or similar if needed, 
        # but normally dictionary iteration is fast enough for 80k.
        for word, count in self._sym_spell.words.items():
            if word.startswith(prefix):
                matches.append((word, count))
                if len(matches) > 100: # Cap search for speed
                    break
        
        # Rank by: (Exact match priority, Frequency count)
        # matches.sort(key=lambda x: (x[0].lower() == prefix, x[1]), reverse=True)
        # Actually x[0].lower() == prefix is boolean, we want True (1) first.
        matches.sort(key=lambda x: (x[0].lower() == prefix, x[1]), reverse=True)
        
        # Return Top 3 capitalized
        return [m[0].upper() for m in matches[:3]]

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
        """Flush and finalize word."""
        if self._space_committed or not self.current_word:
            return

        raw_word = self.current_word
        
        # 🟢 EXACT MATCH PRIORITY: If the user signed a valid word, keep it.
        # This prevents "HI" → "HIS"
        if raw_word.lower() in self._sym_spell.words:
            chosen_word = raw_word
        else:
            # 🟡 CORRECTION: If it's not a word, try to fix typos (e.g., "HEILO" → "HELLO")
            # We use 'raw_word' here; _correct_word uses SymSpell's correction logic.
            chosen_word = self._correct_word(raw_word)

        self.sentence += chosen_word + " "
        self.current_word = ""
        self._last_letter = None
        self._space_committed = True


    def _finalise_sentence(self) -> None:
        """Finalize with memory protection to prevent loops."""
        if self.current_word:
            self._commit_word()

        final_text = self.sentence.strip()
        if not final_text:
            return

        # MEMORY PROTECTION: Only trigger if meaningful change
        if final_text != self._last_spoken_sentence:
            print(f"[Sentence] {final_text}")
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