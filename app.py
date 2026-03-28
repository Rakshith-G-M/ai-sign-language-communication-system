"""
app.py
──────
Streamlit UI for the AI Sign Language Communication System.

Wires together:
    - OpenCV webcam capture
    - realtime_asl_predictor.predict_frame()   (letter prediction)
    - TextBuilder                               (word / sentence / speech)

Run with:
    streamlit run app.py
"""

import time
import cv2
import numpy as np
import streamlit as st

from src.inference.realtime_asl_predictor import predict_frame
from src.inference.text_builder import TextBuilder

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sign Language",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background-color: #0a0a0f;
        color: #e8e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #1e1e2e;
    }
    [data-testid="stSidebar"] * {
        color: #c8c8d8 !important;
    }

    /* ── Title ── */
    .app-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: -0.03em;
        color: #e8e8f0;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: #5a5a7a;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    /* ── Feed placeholder ── */
    .feed-placeholder {
        background: #0f0f1a;
        border: 1px dashed #2a2a3e;
        border-radius: 8px;
        height: 360px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #3a3a5a;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
    }

    /* ── Status pill ── */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 100px;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .status-live {
        background: rgba(0,255,120,0.1);
        border: 1px solid rgba(0,255,120,0.3);
        color: #00ff78;
    }
    .status-idle {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        color: #5a5a7a;
    }

    /* ── Output panels ── */
    .panel {
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .panel-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4a4a6a;
        margin-bottom: 0.5rem;
    }
    .word-text {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 2.4rem;
        color: #a78bfa;
        letter-spacing: -0.02em;
        min-height: 3rem;
        line-height: 1.1;
    }
    .sentence-text {
        font-family: 'Syne', sans-serif;
        font-weight: 400;
        font-size: 1.5rem;
        color: #e8e8f0;
        letter-spacing: -0.01em;
        min-height: 2.5rem;
        line-height: 1.3;
    }
    .word-cursor {
        display: inline-block;
        width: 3px;
        height: 0.9em;
        background: #a78bfa;
        margin-left: 3px;
        vertical-align: middle;
        animation: blink 1s step-end infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* ── Sidebar buttons ── */
    .stButton > button {
        width: 100%;
        border-radius: 7px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        padding: 0.6rem 1rem;
        border: none;
        transition: all 0.15s ease;
        margin-bottom: 0.4rem;
    }
    .stButton > button:first-child {
        background: #a78bfa;
        color: #0a0a0f;
        font-weight: 700;
    }
    .stButton > button:first-child:hover {
        background: #c4b5fd;
    }

    /* ── Divider ── */
    hr {
        border-color: #1e1e2e !important;
        margin: 1.2rem 0 !important;
    }

    /* ── Streamlit image caption hide ── */
    [data-testid="stImage"] > div > div { display: none; }

    /* ── Hand status bar ── */
    .hand-status {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.3rem 0;
        margin-bottom: 0.4rem;
    }
    .hand-on  { color: #00ff78; }
    .hand-off { color: #3a3a5a; }

    /* ── FPS badge ── */
    .fps-badge {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        color: #2e2e4a;
        letter-spacing: 0.08em;
        text-align: right;
        margin-top: 0.3rem;
    }

    /* ── Improved placeholder ── */
    .feed-placeholder-inner {
        text-align: center;
        line-height: 2;
    }
    .feed-placeholder-inner .ready-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #3a3a5a;
        letter-spacing: -0.01em;
    }
    .feed-placeholder-inner .ready-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #2a2a42;
        letter-spacing: 0.08em;
    }

    /* ── Sentence animated container ── */
    .sentence-animated {
        transition: opacity 0.2s ease;
        opacity: 0.95;
    }

    /* ── Footer ── */
    .app-footer {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        color: #2a2a3e;
        letter-spacing: 0.1em;
        text-align: center;
        text-transform: uppercase;
        padding: 2rem 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "running":       False,
        "current_word":  "",
        "sentence":      "",
        "text_builder":  None,
        "prev_sentence": "",   # speech-toast dedup
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤟 Controls")
    st.markdown("---")

    if not st.session_state.running:
        if st.button("▶  Start Camera"):
            st.session_state.running = True
            st.session_state.text_builder = TextBuilder()
            st.session_state.current_word = ""
            st.session_state.sentence = ""
            st.rerun()
    else:
        if st.button("⏹  Stop Camera"):
            st.session_state.running = False
            st.rerun()

    if st.button("↺  Reset Text"):
        st.session_state.current_word = ""
        st.session_state.sentence = ""
        if st.session_state.text_builder is not None:
            st.session_state.text_builder.reset()

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Space Mono",monospace;font-size:0.62rem;
                color:#3a3a5a;line-height:1.8;letter-spacing:0.06em;'>
    HAND ABSENT &gt; 1s → WORD BREAK<br>
    HAND ABSENT &gt; 3s → SENTENCE END<br>
    MIN HOLD: 0.6s PER LETTER<br>
    REPEAT COOLDOWN: 1.2s
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">AI Sign Language</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Real-time ASL · MediaPipe · XGBoost · Edge TTS</p>', unsafe_allow_html=True)

# Status pill
if st.session_state.running:
    st.markdown('<span class="status-pill status-live">● Live</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-pill status-idle">○ Idle</span>', unsafe_allow_html=True)

# Two-column layout: feed left, output right
col_feed, col_output = st.columns([3, 2], gap="large")

with col_feed:
    hand_status_slot = st.empty()
    feed_slot        = st.empty()
    fps_slot         = st.empty()

with col_output:
    word_slot     = st.empty()
    sentence_slot = st.empty()

footer_slot = st.empty()
footer_slot.markdown(
    '<div class="app-footer">Built with MediaPipe · XGBoost · Edge TTS</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — render panels
# ─────────────────────────────────────────────────────────────────────────────
def _render_word(slot, word: str) -> None:
    cursor = '<span class="word-cursor"></span>' if word else ""
    slot.markdown(f"""
    <div class="panel">
        <div class="panel-label">Current Word</div>
        <div class="word-text">{word or "&nbsp;"}{cursor}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_sentence(slot, sentence: str) -> None:
    slot.markdown(f"""
    <div class="panel sentence-animated">
        <div class="panel-label">Sentence</div>
        <div class="sentence-text">{sentence.strip() or "&nbsp;"}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_placeholder(slot) -> None:
    slot.markdown("""
    <div class="feed-placeholder">
        <div class="feed-placeholder-inner">
            <div class="ready-title">READY TO START</div>
            <div class="ready-sub">Show your hand to begin recognition</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Idle state render
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.running:
    hand_status_slot.markdown(
        '<div class="hand-status hand-off">⚫ Waiting for hand...</div>',
        unsafe_allow_html=True,
    )
    _render_placeholder(feed_slot)
    fps_slot.markdown('<div class="fps-badge"></div>', unsafe_allow_html=True)
    _render_word(word_slot, st.session_state.current_word)
    _render_sentence(sentence_slot, st.session_state.sentence)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Live prediction loop
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

tb: TextBuilder = st.session_state.text_builder

try:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        # Mirror for natural view
        frame = cv2.flip(frame, 1)

        # ── Prediction ────────────────────────────────────────────────────────
        # predict_frame() returns (annotated_frame, stable_letter, hand_detected)
        annotated, stable_letter, hand_detected = predict_frame(frame)

        # ── TextBuilder update ────────────────────────────────────────────────
        current_word, sentence = tb.update(
            stable_letter,
            hand_detected,
            time.time(),
        )
        st.session_state.current_word = current_word
        st.session_state.sentence     = sentence

        # ── Hand status indicator ─────────────────────────────────────────────
        if hand_detected:
            hand_status_slot.markdown(
                '<div class="hand-status hand-on">🟢 Detecting Hand</div>',
                unsafe_allow_html=True,
            )
        else:
            hand_status_slot.markdown(
                '<div class="hand-status hand-off">⚫ Waiting for hand...</div>',
                unsafe_allow_html=True,
            )

        # ── Display frame (BGR → RGB) ─────────────────────────────────────────
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        feed_slot.image(rgb, channels="RGB", use_container_width=True)

        # ── FPS indicator ─────────────────────────────────────────────────────
        fps_slot.markdown(
            '<div class="fps-badge">FPS: ~30</div>',
            unsafe_allow_html=True,
        )

        # ── Speech toast — fires only when sentence changes ───────────────────
        if sentence and sentence != st.session_state.prev_sentence:
            st.toast("🔊 Speaking...")
            st.session_state.prev_sentence = sentence

        # ── Update text panels ────────────────────────────────────────────────
        _render_word(word_slot, current_word)
        _render_sentence(sentence_slot, sentence)

        time.sleep(0.03)   # ~30 fps ceiling; keeps CPU usage reasonable

finally:
    cap.release()