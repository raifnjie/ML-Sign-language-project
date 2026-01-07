import cv2
import mediapipe as mp
import numpy as np
import pickle

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def load_model():
    """"Loading the trained model here"""
    try:
        with open("models/sign_language_model.pkl", "rb") as f:
            model = pickle.load(f)
        labels = np.load("models/labels.npy", allow_pickle=True)
        print(" Model loaded successfully")
        print(f"Recognizes {len(labels)} signs: {', '.join(labels)}")
        return model, labels
    except FileNotFoundError:
        print("ERROR: Model not found!")
        print("Please run scripts/train_model.py first")
        return None, None

def predict_sign():
    """Real-time sign language recognition"""
    
    model, labels = load_model()
    if model is None:
        return
    
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            prediction_text = "No hand detected"
            confidence_text = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract landmarks for prediction
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Make prediction
                    landmarks = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(landmarks)[0]
                    
                    # Get prediction probabilities
                    probabilities = model.predict_proba(landmarks)[0]
                    confidence = np.max(probabilities) * 100
                    
                    prediction_text = f"Sign: {prediction}"
                    confidence_text = f"Confidence: {confidence:.1f}%"
            
            # Display prediction on frame
            cv2.putText(frame, prediction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_sign()