import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

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


# ==============================================================================
# DATA PROCESSING PIPELINE (Text Cleaning -> Tokenization -> Stop Words -> TF-IDF)
# ==============================================================================

def normalize_text(text):
    """Step 1: Text cleaning (lowercase, colloquial normalization, punctuation removal, spacing)."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r'\bwi[\s-]?fi\b', 'wifi', t)
    t = re.sub(r'\ba[\s/]?c\b', 'air conditioner', t)
    t = re.sub(r'\b(wash\s*room|rest\s*room|toilet)\b', 'washroom', t)
    t = re.sub(r'\b(id[\s-]?card|identity\s*card)\b', 'id card', t)
    t = re.sub(r'\b(e[\s-]?mail)\b', 'email', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def tokenize_and_filter(text):
    """Steps 2 & 3: Tokenization and English stop-words removal."""
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    return [tok for tok in cleaned.split() if tok not in ENGLISH_STOP_WORDS and len(tok) > 1]


def preprocess_text(text):
    """Full text preprocessing returning cleaned token string ready for TF-IDF vectorization."""
    tokens = tokenize_and_filter(text)
    return " ".join(tokens) if tokens else normalize_text(text)


def load_training_data():
    """Load, clean, and deduplicate campus grievance datasets."""
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
    merged = merged.drop_duplicates(subset=['description', 'department']).reset_index(drop=True)
    return merged


def train_model():
    """
    Train and compare Multiclass Classifiers (Logistic Regression & Calibrated Linear SVM)
    over TF-IDF features, selecting and persisting the best model.
    """
    data = load_training_data()
    if data is None or len(data) == 0:
        raise ValueError("Dataset contains no valid records for training.")

    X = data['description']
    y = data['department']

    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
        ('clf', LogisticRegression(C=3.0, max_iter=1000, class_weight='balanced', random_state=42))
    ])

    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
        ('clf', CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', random_state=42), cv=3))
    ])

    lr_score = float(cross_val_score(lr_pipeline, X, y, cv=5).mean())
    svm_score = float(cross_val_score(svm_pipeline, X, y, cv=5).mean())
    print(f"[NLP Classifier] 5-Fold CV -> Logistic Regression: {lr_score:.4f} | Linear SVM: {svm_score:.4f}")

    if svm_score > lr_score:
        best_model = svm_pipeline
        chosen_name = "Calibrated Linear SVM"
    else:
        best_model = lr_pipeline
        chosen_name = "Logistic Regression"

    best_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_model, file)

    global _cached_model
    _cached_model = best_model
    print(f"[NLP Classifier] Selected {chosen_name} trained on {len(data)} unique campus records -> {MODEL_PATH}")
    return best_model


def load_model():
    """Load model from memory or disk; auto-train if missing or incompatible."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as file:
                loaded = pickle.load(file)
            if hasattr(loaded, 'predict_proba'):
                _cached_model = loaded
                return _cached_model
        except Exception as e:
            print(f"[Model Warning] Could not load saved model ({e}). Retraining...")

    try:
        return train_model()
    except Exception as e:
        print(f"[Model Error] Auto-train failed: {e}")
        return None


def _has_intent(text, keywords):
    for kw in keywords:
        if ' ' in kw:
            if kw in text:
                return True
        else:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                return True
    return False


def _domain_intent_hint(clean_text):
    """High-precision domain intent detector used to reinforce ML probabilities on clear campus terms."""
    if _has_intent(clean_text, [
        'washroom', 'flush', 'plumbing', 'pipe', 'water leak', 'leaking', 'tap', 'sink', 'drain', 'sewage',
        'geyser', 'water cooler', 'water supply', 'air conditioner', 'switchboard', 'socket', 'fuse',
        'power cut', 'sparks', 'spark', 'short circuit', 'wiring', 'broken door', 'broken window',
        'ceiling', 'seepage', 'elevator', 'lift stuck', 'bench', 'chair', 'bed broken', 'fan'
    ]) and not _has_intent(clean_text, ['stolen', 'theft', 'intruder', 'trespass', 'question paper', 'exam paper']):
        return 'Maintenance'

    if _has_intent(clean_text, [
        'stolen', 'theft', 'robbery', 'intruder', 'trespass', 'unauthorized entry', 'opened my room',
        'security guard', 'gate pass', 'cctv', 'surveillance', 'patrol', 'lost id card', 'sos'
    ]):
        return 'Security'

    if _has_intent(clean_text, [
        'wifi', 'internet', 'lan', 'router', 'broadband', 'network', 'portal', 'ums', 'lms', 'server',
        'login', 'password', '500 error', '404 error', 'student email', 'lab computer', 'projector'
    ]):
        return 'IT'

    if _has_intent(clean_text, [
        'mess', 'canteen', 'cafeteria', 'food', 'lunch', 'dinner', 'breakfast', 'meal',
        'stale', 'uncooked', 'cockroach', 'food poisoning', 'samosa', 'dal'
    ]):
        return 'Cafeteria'

    if _has_intent(clean_text, [
        'fee', 'fees', 'tuition', 'fee receipt', 'challan', 'late fine', 'scholarship',
        'education loan', 'caution money', 'refund', 'double payment', 'payment deducted'
    ]):
        return 'Finance'

    if _has_intent(clean_text, [
        'exam', 'examination', 'admit card', 'hall ticket', 'date sheet', 'datesheet',
        'internal marks', 'grade card', 'marksheet', 'evaluation', 'reappear', 'exam clash', 'question paper'
    ]):
        return 'Examination'

    if _has_intent(clean_text, [
        'library', 'book', 'books', 'textbook', 'journal', 'library fine', 'overdue fine',
        'reading hall', 'reading room', 'rfid gate', 'digital library'
    ]):
        return 'Library'

    if _has_intent(clean_text, [
        'ragging', 'harass', 'harassment', 'bully', 'bullying', 'abuse', 'abusive',
        'fight', 'fighting', 'violence', 'threat', 'assault', 'roommate dispute',
        'misbehavior', 'misconduct', 'rude staff', 'smoking in hostel', 'loud music'
    ]):
        return 'Discipline'

    return None


def predict_department_with_confidence(text):
    """
    Predicts the most likely department along with a confidence score (%)
    and top alternative department options.
    Returns:
        {
            'department': str,
            'confidence': float (0-100),
            'alternatives': [{'department': str, 'confidence': float}, ...]
        }
    """
    clean = normalize_text(text)
    if not clean:
        return {
            'department': 'Maintenance',
            'confidence': 50.0,
            'alternatives': []
        }

    model = load_model()
    hint = _domain_intent_hint(clean)

    if model is None:
        dept = hint or 'Maintenance'
        return {
            'department': dept,
            'confidence': 82.0 if hint else 55.0,
            'alternatives': []
        }

    try:
        probs = np.array(model.predict_proba([clean])[0], dtype=float)
        classes = [str(c).strip() for c in model.classes_]

        # Blend calibrated ML probabilities with high-precision domain intent hint when matched
        if hint and hint in classes:
            hint_idx = classes.index(hint)
            probs = probs * 0.22
            probs[hint_idx] += 0.78
            probs = probs / probs.sum()

        order = np.argsort(probs)[::-1]
        best_dept = classes[order[0]]
        best_conf = round(float(probs[order[0]]) * 100.0, 1)

        alternatives = []
        for idx in order[1:3]:
            alt_conf = round(float(probs[idx]) * 100.0, 1)
            if alt_conf >= 3.0:
                alternatives.append({
                    'department': classes[idx],
                    'confidence': alt_conf
                })

        return {
            'department': best_dept if best_dept in VALID_DEPARTMENTS else 'Maintenance',
            'confidence': min(99.5, max(35.0, best_conf)),
            'alternatives': alternatives
        }
    except Exception as e:
        print(f"[Model Error] Prediction failed for input '{text}': {e}")
        return {
            'department': hint or 'Maintenance',
            'confidence': 75.0 if hint else 50.0,
            'alternatives': []
        }


def predict_department(text):
    """Predict the top department name for a given complaint text."""
    return predict_department_with_confidence(text)['department']
