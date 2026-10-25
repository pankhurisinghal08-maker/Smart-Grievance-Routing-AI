import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
import os

# This function teaches the AI using your CSV dataset
def train_model():
    # 1. Read the dataset
    data = pd.read_csv('dataset/complaints_train.csv')
    
    # 2. Set up the pipeline (Text to Numbers -> AI Algorithm)
    # TfidfVectorizer changes English words into numbers the computer understands.
    # MultinomialNB is the math algorithm that guesses the category.
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    # 3. Train the AI
    model.fit(data['description'], data['department'])
    
    # 4. Save the trained AI brain to a file
    with open('model/saved_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("AI Model trained and saved!")

# This function uses the saved AI to guess the department for a new complaint
def predict_department(text):
    # Check if the saved brain exists
    if not os.path.exists('model/saved_model.pkl'):
        return "Model not trained yet"
        
    # Load the brain and make a guess
    with open('model/saved_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    prediction = model.predict([text])
    return prediction[0]