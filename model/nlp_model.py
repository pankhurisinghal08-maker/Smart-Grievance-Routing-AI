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
    """Load training data, using the test split only for legacy installs without a train split."""
    dfs = []
    paths = [DATASET_TRAIN] if os.path.exists(DATASET_TRAIN) else [DATASET_TEST]
    for path in paths:
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

    try:
        return train_model()
    except Exception as e:
        print(f"[Model Error] Auto-train failed: {e}")
        return None

def has_intent(text, keywords):
    """Check if any keyword or phrase appears in text with word boundary safety for single words."""
    for kw in keywords:
        if ' ' in kw or '-' in kw:
            if kw in text:
                return True
        else:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                return True
    return False

def predict_department(text):
    """Predict the department for a given complaint text with high-precision semantic intent verification."""
    clean = normalize_text(text)
    if not clean:
        return "Maintenance"

    t = clean.lower()

    # 1. Maintenance Intent (Plumbing, sanitation, electrical, civil, furniture, cleaning)
    # Checked first with high priority even if words like 'warden' appear
    if has_intent(t, [
        'toilet', 'washroom', 'flush', 'plumb', 'plumbing', 'pipe', 'leak', 'leakage', 'tap', 'sink', 'drain', 'sewage',
        'geyser', 'cooler', 'water supply', 'dirty water', 'water filter', 'choke', 'blocked',
        'clean toilet', 'cleaning toilet', 'bathroom', 'sanitation', 'restroom',
        'ac not', 'air condition', 'switchboard', 'socket', 'fuse', 'power cut', 'spark', 'sparks',
        'light not', 'fan not', 'bulb', 'wiring', 'short circuit',
        'broken door', 'broken window', 'ceiling', 'seepage', 'elevator', 'lift stuck',
        'furniture', 'bench', 'chair', 'bed broken', 'pest control', 'mosquito'
    ]):
        if not has_intent(t, ['stolen', 'theft', 'someone opened', 'intruder', 'trespass']):
            return 'Maintenance'

    # 2. Security Intent (Unauthorized entry, stolen items, guard, CCTV, gate)
    if has_intent(t, [
        'opened my room', 'without permission', 'entered my room', 'intruder', 'trespass',
        'unauthorized entry', 'unauthorized access', 'stranger in hostel', 'broke into my room',
        'someone entered', 'room privacy', 'unlocked my room', 'unlocked door',
        'stolen', 'theft', 'robbery', 'someone stole', 'lost laptop', 'stolen phone', 'stolen bicycle',
        'cycle stolen', 'wallet stolen', 'bag stolen', 'locker broken', 'missing belongings',
        'security guard', 'gate security', 'gate pass', 'cctv', 'surveillance', 'frisking',
        'dark streetlight', 'street light on campus road', 'patrol', 'lost id card', 'sos'
    ]):
        return 'Security'

    # 3. IT Intent (WiFi, internet, portals, UMS, LMS, servers, computers)
    if has_intent(t, [
        'wifi', 'wi-fi', 'internet', 'lan', 'router', 'broadband', 'network', 'ip address',
        'portal', 'ums', 'lms', 'server', 'login', 'password', 'reset password', 'credential',
        '500 error', '404 error', 'website down', 'student email', 'biometric attendance error',
        'lab computer', 'pc in lab', 'projector in lab', 'software install'
    ]):
        return 'IT'

    # 4. Cafeteria Intent (Mess, food, canteen, meals, hygiene)
    if has_intent(t, [
        'mess', 'canteen', 'cafeteria', 'food', 'lunch', 'dinner', 'breakfast', 'meal',
        'stale food', 'uncooked food', 'cockroach in food', 'insect in food', 'hair in food',
        'food poisoning', 'sour milk', 'food hygiene', 'mess menu', 'canteen vendor',
        'canteen charged twice', 'mess food'
    ]):
        return 'Cafeteria'

    # 5. Finance Intent (Fees, refunds, receipts, challan, accounts)
    if has_intent(t, [
        'fee', 'fees', 'tuition', 'fee receipt', 'fee challan', 'late fine', 'fine waiver',
        'scholarship', 'education loan', 'caution money', 'security deposit refund',
        'payment deducted', 'double payment', 'bank transaction', 'account balance', 'utr'
    ]):
        return 'Finance'

    # 6. Examination Intent (Exams, marks, admit card, hall ticket)
    if has_intent(t, [
        'exam', 'examination', 'admit card', 'hall ticket', 'date sheet', 'datesheet',
        'internal marks', 'grade card', 'marksheet', 're-evaluation', 'reappear',
        'mid term', 'end term', 'exam clash', 'scribes'
    ]):
        return 'Examination'

    # 7. Library Intent (Books, circulation, library fine)
    if has_intent(t, [
        'library', 'book', 'textbook', 'journal', 'library fine', 'overdue fine',
        'borrow book', 'return book', 'reading hall', 'reading room', 'rfid gate',
        'digital library', 'ieee'
    ]):
        return 'Library'

    # 8. Discipline Intent (Staff conduct, hospital/dispensary, proctorial, ragging, harassment, misconduct)
    if has_intent(t, [
        'hospital', 'hospital staff', 'dispensary', 'clinic', 'medical staff', 'doctor', 'nurse', 'health center',
        'staff is', 'staff are', 'staff was', 'staff were', 'rude staff', 'annoying staff', 'staff misbehavior', 'staff misconduct',
        'annoying', 'rude', 'arrogant', 'unprofessional', 'misbehavior', 'misconduct', 'bad behavior', 'uncooperative',
        'ragging', 'harass', 'harassment', 'bully', 'bullying', 'abuse', 'abusive',
        'fight', 'fighting', 'violence', 'threat', 'threatening', 'assault',
        'loud music at night', 'shouting in corridor', 'smoking in hostel', 'alcohol',
        'drugs in room', 'roommate dispute', 'quarrel', 'defamation', 'eve teasing'
    ]):
        return 'Discipline'

    # Fallback to Trained Machine Learning Model
    model = load_model()
    if model is not None:
        try:
            prediction = model.predict([clean])
            predicted_dept = str(prediction[0]).strip()
            if predicted_dept in VALID_DEPARTMENTS:
                return predicted_dept
        except Exception as e:
            print(f"[Model Error] Prediction failed for input '{text}': {e}")

    return "Maintenance"
