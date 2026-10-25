import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
import os

def train_model():
    # Read your training data
    data = pd.read_csv('dataset/complaints_train.csv')
    
    # Build and train the AI
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(data['description'], data['department'])
    
    # Make sure the model folder exists, then save the AI brain
    os.makedirs('model', exist_ok=True)
    with open('model/saved_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("AI Model trained and saved!")

def predict_department(text):
    # Check if AI brain file exists
    if not os.path.exists('model/saved_model.pkl'):
        return "Model not trained yet"
        
    # Open AI brain and make a guess
    with open('model/saved_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    prediction = model.predict([text])
    return prediction[0]