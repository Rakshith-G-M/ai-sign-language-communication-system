import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function PredictionPanel({ letter, confidence }) {
  return (
    <motion.div
      className={`prediction-panel ${letter ? 'active' : ''}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <AnimatePresence mode="wait">
        {letter ? (
          <motion.div
            key={letter}
            initial={{ opacity: 0, scale: 0.5, rotateZ: -10 }}
            animate={{ opacity: 1, scale: 1, rotateZ: 0 }}
            exit={{ opacity: 0, scale: 0.5, rotateZ: 10 }}
            transition={{ duration: 0.3, type: 'spring', stiffness: 200 }}
          >
            <h1 className="letter-display detected">{letter}</h1>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h1 className="letter-display empty">?</h1>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="confidence-bar">
        <motion.div
          className="confidence-fill"
          initial={{ width: 0 }}
          animate={{ width: `${confidence * 100}%` }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
        />
      </div>

      <p className="prediction-label">
        {letter ? `Confidence: ${(confidence * 100).toFixed(0)}%` : 'Waiting for hand...'}
      </p>
    </motion.div>
  );
}

export default PredictionPanel;