# AI Sign Language Communication System

A real-time AI system that converts **sign language hand gestures into text and speech** using computer vision and machine learning.

This project aims to reduce the communication barrier between **deaf and hearing individuals** by enabling a system that can understand sign language through a webcam and translate it into readable and spoken language.

---

## Project Overview

The system detects hand gestures using **MediaPipe and OpenCV**, extracts hand landmark features, and uses a machine learning model to classify gestures. The recognized gesture is then converted into **text output and optional speech output**.

---

## Features

- Real-time hand landmark detection
- Sign language gesture recognition
- Gesture to text conversion
- Text-to-speech output
- Webcam-based interaction
- Modular and scalable project architecture

---

## Tech Stack

### Programming Language
- Python

### Libraries and Frameworks
- OpenCV
- MediaPipe
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

```
ai-sign-language-communication-system
│
├── src
│   └── vision
│       └── hand_detector.py
│
├── dataset
│
├── models
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/ai-sign-language-communication-system.git
```

### 2. Navigate to the project directory

```
cd ai-sign-language-communication-system
```

### 3. Create a virtual environment

```
python -m venv venv
```

### 4. Activate the virtual environment

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

### 5. Install dependencies

```
pip install -r requirements.txt
```

---

## Development Roadmap

### Stage 1
Environment setup and repository initialization

### Stage 2
Hand landmark detection using MediaPipe and OpenCV

### Stage 3
Gesture dataset creation

### Stage 4
Machine learning model training for gesture classification

### Stage 5
Real-time gesture recognition system

### Stage 6
Text-to-speech integration

---

## Future Improvements

- Support for a larger sign language vocabulary
- Deep learning based gesture recognition
- Mobile or web application deployment
- Multi-language speech output
- Real-time sentence generation

---

## Author

Rakshith G M  
BCA Student | AI & Computer Vision Enthusiast  

GitHub:  
https://github.com/Rakshith-web-dev

---

## License

This project is developed for educational and research purposes.