import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Base directory for resolving relative paths reliably
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'saved_model.pkl')
DATASET_TRAIN = os.path.join(BASE_DIR, 'dataset', 'complaints_train.csv')
DATASET_TEST = os.path.join(BASE_DIR, 'dataset', 'complaints_test.csv')

_cached_model = None

def get_dataset_path():
    """Find available training dataset file."""
    if os.path.exists(DATASET_TRAIN):
        return DATASET_TRAIN
    if os.path.exists(DATASET_TEST):
        return DATASET_TEST
    return None

def train_model():
    """Read dataset, train TF-IDF + MultinomialNB pipeline, and save to pickle."""
    dataset_path = get_dataset_path()
    if not dataset_path:
        raise FileNotFoundError(
            f"No training dataset found at '{DATASET_TRAIN}' or '{DATASET_TEST}'."
        )

    # Read and sanitize training data
    data = pd.read_csv(dataset_path)
    data = data.dropna(subset=['description', 'department'])
    data['description'] = data['description'].astype(str).str.strip()
    data['department'] = data['department'].astype(str).str.strip()
    data = data[(data['description'] != '') & (data['department'] != '')]

    if len(data) == 0:
        raise ValueError("Dataset contains no valid records for training.")

    # Build and train the pipeline
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True),
        MultinomialNB(alpha=0.1)
    )
    model.fit(data['description'], data['department'])

    # Save trained model to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    global _cached_model
    _cached_model = model
    print(f"AI Model trained successfully on {len(data)} samples and saved to {MODEL_PATH}")
    return model

def load_model():
    """Load model from memory or disk; auto-train if not yet built."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as file:
                _cached_model = pickle.load(file)
            return _cached_model
        except Exception as e:
            print(f"[Model Warning] Could not load saved model ({e}). Retraining...")

    # Auto-train if saved model doesn't exist or failed to load
    try:
        return train_model()
    except Exception as e:
        print(f"[Model Error] Auto-train failed: {e}")
        return None

def predict_department(text):
    """Predict the department for a given complaint text."""
    if not text or not str(text).strip():
        return "General"

    model = load_model()
    if model is None:
        return "General"

    try:
        prediction = model.predict([str(text).strip()])
        return prediction[0]
    except Exception as e:
        print(f"[Model Error] Prediction failed for input '{text}': {e}")
        return "General"
