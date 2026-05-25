import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/Resume.csv")

# Text cleaning function
def clean_resume(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    filtered_words = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(filtered_words)

# Clean resumes
df['cleaned_resume'] = df['Resume_str'].apply(clean_resume)

# Features and labels
X = df['cleaned_resume']
y = df['Category']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

X_tfidf = tfidf.fit_transform(X)

# Train model
model = LinearSVC()

model.fit(X_tfidf, y)

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

# Save TF-IDF vectorizer
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Model and TF-IDF saved successfully")