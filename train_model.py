import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def load_data():
    """Load all collected sign data"""
    X = []  # Features (landmark coordinates)
    y = []  # Labels (sign names)
    
    data_dir = 'data'
    
    # Get all .npy files
    sign_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if len(sign_files) == 0:
        print("ERROR: No data files found in 'data/' folder")
        print("Please run collect_data.py first!")
        return None, None, None
    
    print(f"Found {len(sign_files)} sign(s) to train on:")
    
    for sign_file in sign_files:
        # Get sign name (remove .npy extension)
        sign_name = sign_file[:-4]
        
        # Load data
        sign_data = np.load(os.path.join(data_dir, sign_file))
        
        print(f"  - {sign_name}: {len(sign_data)} samples")
        
        # Add to dataset
        X.extend(sign_data)
        y.extend([sign_name] * len(sign_data))
    
    return np.array(X), np.array(y), sign_files

def train_model():
    """Train a Random Forest classifier on the collected data"""
    
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    
    X, y, sign_files = load_data()
    
    if X is None:
        return
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]} (21 landmarks × 3 coordinates)")
    
    # Split data: 80% training, 20% testing
    print("\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train Random Forest Classifier
    print("\n" + "="*50)
    print("Training Random Forest Classifier...")
    print("="*50)
    
    clf = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Maximum depth of trees
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating model...")
    print("="*50)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Test Accuracy: {accuracy * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = 'models/sign_language_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    # Also save the class labels
    labels_path = 'models/labels.npy'
    np.save(labels_path, clf.classes_)
    print(f"✓ Labels saved to: {labels_path}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)

if __name__ == "__main__":
    train_model()