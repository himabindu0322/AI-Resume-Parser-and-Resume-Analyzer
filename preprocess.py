import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/Resume.csv")

# Display dataset columns
print("Dataset Columns:")
print(df.columns)

# Select first resume text
text = df['Resume_str'][0]

print("\nOriginal Resume Text:\n")
print(text[:1000])  # Print first 1000 characters

# Convert text to lowercase
text = text.lower()

# Remove URLs
text = re.sub(r'http\S+', '', text)

# Remove email addresses
text = re.sub(r'\S+@\S+', '', text)

# Remove special characters and numbers
text = re.sub(r'[^a-zA-Z\s]', '', text)

# Remove extra spaces
text = re.sub(r'\s+', ' ', text).strip()

# Tokenization
tokens = word_tokenize(text)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_words = [
    word for word in tokens
    if word not in stop_words and len(word) > 2
]

print("\nProcessed Tokens:\n")
print(filtered_words[:100])