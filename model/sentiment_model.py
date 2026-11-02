import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'feedback_sentiment_model.pkl')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'feedback_sentiment_train.csv')

_sentiment_pipeline = None

def clean_feedback_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_sentiment_model(csv_path=DATASET_PATH, save_path=MODEL_PATH):
    global _sentiment_pipeline
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df['clean_text'] = df['feedback_text'].apply(clean_feedback_text)
    
    X = df['clean_text']
    y = df['sentiment']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ('clf', LogisticRegression(C=2.0, max_iter=500, random_state=42))
    ])

    pipeline.fit(X, y)
    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"[Sentiment Model] Cross-validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    _sentiment_pipeline = pipeline
    print(f"[Sentiment Model] Model successfully saved to {save_path}")
    return pipeline

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                _sentiment_pipeline = pickle.load(f)
                return _sentiment_pipeline
        except Exception as e:
            print(f"[Sentiment Model] Error loading cached model: {e}")

    # Train if not found
    return train_sentiment_model()

def predict_feedback_sentiment(text):
    """
    Predict whether student resolution feedback is Positive or Negative.
    Returns: ('Positive' or 'Negative', confidence_score)
    """
    if not text or not text.strip():
        return 'Negative', 0.0

    cleaned = clean_feedback_text(text)
    if not cleaned:
        return 'Negative', 0.0

    # Rule-based heuristics for explicit critical phrases
    negative_indicators = [
        'not fixed', 'not resolved', 'still leaking', 'still broken', 'still not',
        'unresolved', 'not satisfied', 'did not show up', 'nobody came', 'terrible',
        'worst', 'poor service', 'error persists', 'still showing', 'disappointed',
        'dissatisfied', 'useless', 'not working', 'ignored', 'incomplete'
    ]
    positive_indicators = [
        'resolved satisfactorily', 'thank you', 'excellent', 'great job', 'working fine',
        'working perfectly', 'satisfied', 'promptly', 'appreciate', 'fixed', 'good work',
        'feeling safe', 'completely satisfied', 'well done', 'speedy resolution'
    ]

    for ind in negative_indicators:
        if ind in cleaned:
            return 'Negative', 0.95

    pipeline = get_sentiment_pipeline()
    try:
        prediction = pipeline.predict([cleaned])[0]
        probs = pipeline.predict_proba([cleaned])[0]
        pos_idx = list(pipeline.classes_).index('Positive')
        confidence = float(probs[pos_idx])
        return prediction, confidence
    except Exception as e:
        print(f"[Sentiment Model] Prediction error: {e}")
        # Fallback to keyword matching
        if any(ind in cleaned for ind in positive_indicators):
            return 'Positive', 0.85
        return 'Negative', 0.50

if __name__ == '__main__':
    pipe = train_sentiment_model()
    test_samples = [
        "Issue resolved satisfactorily, thank you so much!",
        "The problem is still not fixed, technician never visited.",
        "Wifi is working great now.",
        "Terrible experience, water still leaking from ceiling."
    ]
    print("\n--- Test Predictions ---")
    for s in test_samples:
        pred, conf = predict_feedback_sentiment(s)
        print(f"'{s}' -> {pred} ({conf:.2f})")
