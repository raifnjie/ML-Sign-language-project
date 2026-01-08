# Sign Language Recognition

A real-time sign language recognition system using MediaPipe, OpenCV, and scikit-learn. This project tracks keypoints on hands live, using mediapipe in order to read sign langauge gestures in real time!

## Features
- Hand landmark detection using MediaPipe
- Custom sign language gesture training
- Real-time prediction with confidence scores

## Installation
```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/sign-language-recognition.git
cd sign-language-recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install mediapipe opencv-python scikit-learn numpy
```

## Usage

### 1. Collect Training Data
```bash
python scripts/collect_data.py
```
- Press 's' to start collecting samples for each sign
- Press 'q' to skip a sign

### 2. Train the Model
```bash
python scripts/train_model.py
```

### 3. Run Real-Time Recognition
```bash
python main.py
```
- Press 'q' to quit

## Project Structure
```
sign_language_project/
├── data/              # Training data
├── models/            # Trained models
├── scripts/
│   ├── collect_data.py
│   └── train_model.py
└── main.py            # Real-time prediction
```

## Technologies Used
- **MediaPipe**: Hand landmark detection
- **OpenCV**: Video capture and display
- **scikit-learn**: Machine learning (Random Forest Classifier)
- **NumPy**: Data processing
