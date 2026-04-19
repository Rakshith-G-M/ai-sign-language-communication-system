import React from 'react';
import { motion } from 'framer-motion';

function ControlBar({ isRunning, onStartStop, onReset }) {
  return (
    <motion.div
      className="control-bar"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <motion.button
        className={`control-btn btn-primary ${isRunning ? 'running' : ''}`}
        onClick={onStartStop}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <span>{isRunning ? '⏹ STOP' : '▶ START'}</span>
      </motion.button>

      <motion.button
        className="control-btn btn-secondary"
        onClick={onReset}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <span>↻ RESET</span>
      </motion.button>
    </motion.div>
  );
}

export default ControlBar;