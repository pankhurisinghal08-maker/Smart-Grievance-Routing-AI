from model.nlp_model import train_model
from model.sentiment_model import train_sentiment_model

print("1. Training Department Routing NLP Classifier...")
train_model()

print("\n2. Training Student Feedback Sentiment Classifier...")
train_sentiment_model()

print("\nAll AI models successfully trained and persisted.")
