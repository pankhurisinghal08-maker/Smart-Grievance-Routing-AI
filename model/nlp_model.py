import os
import re
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

VALID_DEPARTMENTS = [
    'IT',
    'Maintenance',
    'Cafeteria',
    'Security',
    'Finance',
    'Library',
    'Discipline',
    'Examination'
]
_cached_model = None

def normalize_text(text):
    """Normalize text variations so colloquial terms match dataset tokens."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r'\bwi[\s-]?fi\b', 'wifi', t)
    t = re.sub(r'\ba[\s/]?c\b', 'air conditioner', t)
    t = re.sub(r'\b(wash\s*room|rest\s*room|toilet)\b', 'washroom', t)
    t = re.sub(r'\b(id[\s-]?card|identity\s*card)\b', 'id card', t)
    t = re.sub(r'\b(e[\s-]?mail)\b', 'email', t)
    return t.strip()

def load_training_data():
    """Load and merge all available training and test datasets for optimal accuracy."""
    dfs = []
    for path in [DATASET_TRAIN, DATASET_TEST]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"[Model Warning] Could not read {path}: {e}")
    if not dfs:
        return None
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=['description', 'department'])
    merged['description'] = merged['description'].astype(str).map(normalize_text)
    merged['department'] = merged['department'].astype(str).str.strip()
    merged = merged[(merged['description'] != '') & (merged['department'].isin(VALID_DEPARTMENTS))]
    return merged

def train_model():
    """Read all datasets, train TF-IDF + MultinomialNB pipeline, and save to pickle."""
    data = load_training_data()
    if data is None or len(data) == 0:
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
    print(f"AI Model trained successfully on {len(data)} campus records and saved to {MODEL_PATH}")
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
    """Predict the department for a given complaint text, guaranteeing a valid campus department."""
    clean = normalize_text(text)
    if not clean:
        return "Maintenance"

    model = load_model()
    if model is None:
        # Heuristic fallback if model unavailable
        if any(w in clean for w in ['exam', 'examination', 'marksheet', 'grade card', 'hall ticket', 'admit card', 're-evaluation', 'reappear']):
            return 'Examination'
        if any(w in clean for w in ['wifi', 'laptop', 'portal', 'pc', 'login', 'email', 'network']):
            return 'IT'
        if any(w in clean for w in ['food', 'canteen', 'lunch', 'counter', 'coffee', 'snack']):
            return 'Cafeteria'
        if any(w in clean for w in ['guard', 'gate', 'theft', 'stolen', 'suspicious', 'vehicle']):
            return 'Security'
        if any(w in clean for w in ['fee', 'receipt', 'refund', 'fine', 'scholarship', 'finance']):
            return 'Finance'
        if any(w in clean for w in ['book', 'library', 'catalog', 'reading room']):
            return 'Library'
        if any(w in clean for w in ['fight', 'ragging', 'abuse', 'harass', 'discipline', 'misbehavior']):
            return 'Discipline'
        return "Maintenance"

    try:
        prediction = model.predict([clean])
        predicted_dept = str(prediction[0]).strip()
        if predicted_dept in VALID_DEPARTMENTS:
            return predicted_dept
        return "Maintenance"
    except Exception as e:
        print(f"[Model Error] Prediction failed for input '{text}': {e}")
        return "Maintenance"
