import React from 'react';
import { motion } from 'framer-motion';

function WordBuilder({ word, sentence, suggestions = [], onSpeak }) {
  return (
    <motion.div
      className="word-builder"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      {/* Suggestions Bar (PREDICTIVE STATE) */}
      {suggestions.length > 0 && (
        <div className="suggestions-section">
          <div className="suggestion-label">Predictive Suggestions</div>
          <div className="suggestions-list">
            {suggestions.map((s, i) => (
              <motion.span
                key={s}
                className="suggestion-item"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
              >
                {s}
              </motion.span>
             ))}
          </div>
        </div>
      )}

      {/* Current Word Section (BUILDING STATE) */}
      <div className="word-section">
        <div className="word-label">Active Word</div>
        <motion.div
          className={`word-text ${word ? '' : 'empty'}`}
          key={word}
          initial={{ opacity: 0.5, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2 }}
        >
          {word ? `${word}_` : 'Spell a word...'}
        </motion.div>
      </div>

      {/* Sentence Section (FINALIZED STATE) */}
      <div className="sentence-section">
        <div className="sentence-header">
          <div className="word-label">Committed Sentence</div>
          {sentence && (
            <motion.button
              className="speak-btn"
              onClick={onSpeak}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              🔊 Speak
            </motion.button>
          )}
        </div>
        <motion.div
          className={`sentence-text ${sentence ? '' : 'empty'}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          {sentence || 'Your complete sentence will appear here...'}
        </motion.div>
      </div>
    </motion.div>
  );
}

export default WordBuilder;