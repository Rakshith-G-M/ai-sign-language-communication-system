# AI Sign Language Communication System

A real-time AI system that converts **sign language gestures into text and speech** using computer vision and machine learning.

This project aims to bridge the communication gap between **deaf and hearing individuals** by enabling real-time sign language recognition using a webcam.

---

## Project Overview

The system detects hand gestures using **MediaPipe and OpenCV**, classifies the gesture using **machine learning**, and converts the recognized sign into **text and speech output**.

---

## Features

- Real-time hand landmark detection
- Sign language gesture recognition
- Gesture to text conversion
- Text to speech output
- Webcam-based interaction

---

## Tech Stack

**Programming Language**
- Python

**Libraries & Frameworks**
- OpenCV
- MediaPipe
- Scikit-learn
- Streamlit
- NumPy

---

## Project Structure
ai-sign-language-communication-system
│
├── src/
│ └── vision/
│ └── hand_detector.py
│
├── dataset/
│
├── models/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## Installation

Clone the repository:


git clone https://github.com/YOUR_USERNAME/ai-sign-language-communication-system.git


Navigate into the project directory:


cd ai-sign-language-communication-system


Create virtual environment:


python -m venv venv


Activate environment:

Windows


venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


---

## Development Roadmap

Stage 1  
Environment setup and project structure

Stage 2  
Hand landmark detection using MediaPipe

Stage 3  
Gesture dataset creation

Stage 4  
Machine learning model for gesture classification

Stage 5  
Real-time gesture recognition

Stage 6  
Text-to-speech integration

---

## Future Improvements

- Support for full ASL vocabulary
- Deep learning based gesture recognition
- Mobile application integration
- Multi-language speech output

---

## Author

Rakshith G M

BCA Student | AI & Computer Vision Enthusiast

GitHub  
https://github.com/Rakshith-web-dev

---

## License

This project is developed for educational and research purposes.