import cv2
import mediapipe as mp
import numpy as np
import os 
import time 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def collect_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    # Define the signs you want to collect
    # Start with just a few, you can add more later
    signs = ['A', 'B', 'C', 'Hello', 'Thanks', 'Yes', 'No']
    
    samples_per_sign = 100  # Collect 100 samples per sign
    
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1  # Only track one hand for simplicity
    ) as hands:
        
        for sign in signs:
            print(f"\n{'='*50}")
            print(f"Collecting data for sign: {sign}")
            print(f"{'='*50}")
            print("Press 's' to start collecting")
            print("Press 'q' to skip this sign")
            
            collected_samples = 0
            collecting = False
            sign_data = []
            
            while collected_samples < samples_per_sign:
                success, frame = cap.read()
                if not success:
                    continue
                
                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # If collecting, extract landmarks
                        if collecting:
                            # Extract x, y, z coordinates for all 21 landmarks
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.extend([landmark.x, landmark.y, landmark.z])
                            
                            sign_data.append(landmarks)
                            collected_samples += 1
                
                # Display info on frame
                if collecting:
                    cv2.putText(frame, f"COLLECTING: {sign}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Samples: {collected_samples}/{samples_per_sign}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Ready for: {sign}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Press 's' to start", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Collect Sign Language Data', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and not collecting:
                    collecting = True
                    print("Started collecting...")
                elif key == ord('q'):
                    print(f"Skipped sign: {sign}")
                    break
            
            # Save collected data for this sign
            if len(sign_data) > 0:
                np.save(f'data/{sign}.npy', np.array(sign_data))
                print(f"✓ Saved {len(sign_data)} samples for '{sign}'")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*50)
    print("Data collection complete!")
    print("="*50)

if __name__ == "__main__":
    collect_data()