# AI Sign Language Communication System

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00BCD4?style=flat-square&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Educational%20%26%20Research-22C55E?style=flat-square)

<br>

![Banner](assets/banner.png)

---

## Overview

The **AI Sign Language Communication System** is a real-time computer vision pipeline that translates hand gestures from American Sign Language (ASL) into text and synthesized speech. Designed for accessibility and inclusivity, the system processes live webcam input, classifies gestures using a trained machine learning model, and surfaces results through an interactive Streamlit interface — with no cloud dependency required.

The project targets seamless, low-latency communication assistance for the Deaf and hard-of-hearing community, and serves as an extensible research platform for gesture recognition development.

---

## Demo

![Demo](assets/demo.gif)

> Live gesture recognition running at 30+ FPS via webcam. Detected signs are transcribed to text in real time and optionally converted to speech output.

---

## System Architecture

![System Architecture](assets/system_architecture.png)

The system is composed of three loosely coupled stages: **input capture**, **gesture inference**, and **output rendering**. Each stage is independently replaceable, enabling straightforward experimentation with alternative models or output modalities.

---

## Pipeline Overview

| Stage | Component | Description |
|---|---|---|
| 1. Input | Webcam / Video Feed | Raw RGB frames captured via OpenCV |
| 2. Detection | MediaPipe Hands | 21-point hand landmark extraction per frame |
| 3. Feature Engineering | `hand_detector.py` | Landmark normalization and feature vector construction |
| 4. Classification | Scikit-Learn Model | Trained classifier maps feature vectors to gesture labels |
| 5. Post-processing | Sequence Buffer | Temporal smoothing to reduce flickering predictions |
| 6. Output | Streamlit + TTS | Transcribed text displayed; optional text-to-speech synthesis |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core implementation |
| Computer Vision | OpenCV 4.x | Frame capture, preprocessing, annotation |
| Landmark Detection | MediaPipe Hands | Real-time hand keypoint estimation |
| ML Classification | Scikit-Learn | Gesture classifier training and inference |
| Application UI | Streamlit | Interactive web-based frontend |
| Text-to-Speech | pyttsx3 / gTTS | Audible output from recognized gestures |
| Data Handling | NumPy, Pandas | Feature processing and dataset management |

---

## Project Structure

```
ai-sign-language-communication-system/
│
├── assets/
│   ├── banner.png                  # Project banner
│   ├── system_architecture.png     # Architecture diagram
│   └── demo.gif                    # Recorded demo
│
├── src/
│   └── vision/
│       └── hand_detector.py        # Landmark extraction and feature engineering
│
├── dataset/                        # Training data (gesture samples)
│
├── models/                         # Serialized trained model artifacts
│
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- A connected webcam
- pip package manager

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/Rakshith-web-dev/ai-sign-language-communication-system.git
cd ai-sign-language-communication-system
```

**2. Create and activate a virtual environment**

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Application

**Launch the Streamlit interface**

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`. Grant webcam permissions when prompted.

**Run gesture detection directly (headless)**

```bash
python src/vision/hand_detector.py
```

---

## Development Roadmap

| Milestone | Status | Description |
|---|---|---|
| Real-time hand landmark detection | Complete | MediaPipe integration with 21-keypoint extraction |
| Gesture classification model | Complete | Scikit-Learn classifier trained on ASL dataset |
| Streamlit application UI | Complete | Live feed with transcription overlay |
| Text-to-speech output | Complete | Synthesized speech from recognized gestures |
| Multi-hand support | In Progress | Simultaneous detection of both hands |
| Expanded gesture vocabulary | In Progress | Extending beyond static ASL alphabet to dynamic signs |
| Mobile / edge deployment | Planned | ONNX export for on-device inference |
| Multilingual sign language support | Planned | BSL, ISL, and additional gesture sets |

---

## Future Improvements

- **Dynamic Gesture Recognition** — Current classification handles static poses. Extending the pipeline with an LSTM or Transformer-based sequence model will enable recognition of motion-dependent signs.
- **Custom Vocabulary Training** — A guided data collection workflow will allow users to record and train custom gestures without modifying core pipeline code.
- **Edge Deployment** — Exporting the trained model to ONNX or TFLite will enable inference on mobile devices and embedded systems (e.g., Raspberry Pi).
- **Sentence-Level Context** — Integrating a lightweight language model for post-processing recognized gesture sequences into grammatically coherent sentences.
- **Accessibility API** — A REST API layer to allow third-party applications to consume gesture predictions programmatically.

---

## Author

**Rakshith G M**
Software Engineer — Computer Vision & ML Systems

[![GitHub](https://img.shields.io/badge/GitHub-Rakshith--web--dev-181717?style=flat-square&logo=github)](https://github.com/Rakshith-web-dev)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rakshith%20G%20M-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/rakshith-g-m/)


---

## License

This project is intended for educational and research purposes only. It is not licensed for commercial use or redistribution. If you use this work in academic research, please provide appropriate attribution.

---

<p align="center">Built to make communication more accessible.</p>
