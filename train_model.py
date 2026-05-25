import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/Resume.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Text cleaning function
def clean_resume(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and short words
    filtered_words = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(filtered_words)

# Clean all resumes
df['cleaned_resume'] = df['Resume_str'].apply(clean_resume)

print("\nSample Cleaned Resume:\n")
print(df['cleaned_resume'][0][:1000])

# Features and Labels
X = df['cleaned_resume']
y = df['Category']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_tfidf = tfidf.fit_transform(X)

print("\nTF-IDF Shape:")
print(X_tfidf.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Train SVM Model
model = LinearSVC()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test Custom Resume
sample_resume = """
Python developer with machine learning,
data science, NLP, TensorFlow,
Flask, FastAPI, SQL, and deep learning experience.
"""

# Clean sample resume
cleaned_sample = clean_resume(sample_resume)

# Convert to TF-IDF
sample_vector = tfidf.transform([cleaned_sample])

# Predict category
prediction = model.predict(sample_vector)

print("\nPredicted Resume Category:")
print(prediction[0])