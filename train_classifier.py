
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

def load_data(data_dir):
    """Loads embeddings and labels from the training data directory."""
    embeddings = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith("_embedding.npy"):
                    embedding_path = os.path.join(label_dir, file)
                    embeddings.append(np.load(embedding_path).flatten())
                    labels.append(label)
    return np.array(embeddings), np.array(labels)

def main():
    """Main function to train and save the classifier."""
    logging.info("Starting classifier training...")
    
    X, y = load_data("training_data")
    
    if len(X) == 0:
        logging.error("No training data found. Please run fetch_and_run.py to generate data.")
        return

    logging.info(f"Loaded {len(X)} samples.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info("Training a Logistic Regression classifier...")
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)
    
    logging.info("Evaluating classifier...")
    y_pred = classifier.predict(X_test)
    logging.info("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    logging.info("Saving trained model to land_use_classifier.pkl...")
    joblib.dump(classifier, "land_use_classifier.pkl")
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    main()
