import streamlit as st
import re
import nltk
import pdfplumber
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(
    page_title="AI Resume Parser",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("📄 AI Resume Parser")

st.sidebar.info(
    """
    Upload a PDF resume and get:
    
    ✅ Resume Category Prediction
    
    ✅ Skill Extraction
    
    ✅ Resume Score
    
    ✅ AI Recommendations
    """
)

# -----------------------------------
# Download NLTK Resources
# -----------------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# -----------------------------------
# Load Saved Model & TF-IDF
# -----------------------------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# -----------------------------------
# Text Cleaning Function
# -----------------------------------
def clean_resume(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove Emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove Special Characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove Extra Spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords
    stop_words = set(stopwords.words('english'))

    # Remove Stopwords
    filtered_words = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(filtered_words)

# -----------------------------------
# Skills Database
# -----------------------------------
skills = [

    # Programming Languages
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "typescript",

    # Frontend
    "html",
    "css",
    "react",
    "redux",
    "tailwind",
    "bootstrap",

    # Backend
    "nodejs",
    "express",
    "django",
    "flask",
    "fastapi",

    # Database
    "mysql",
    "postgresql",
    "mongodb",
    "firebase",
    "sql",

    # AI / ML
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "nlp",
    "data science",
    "tensorflow",
    "pytorch",

    # Tools
    "git",
    "github",
    "docker",
    "kubernetes",
    "postman",

    # Cloud
    "aws",
    "azure",
    "cloud computing",

    # Concepts
    "web development",
    "system design",
    "problem solving",
    "rest api",
    "data structures",
    "algorithms"
]

# -----------------------------------
# Main Title
# -----------------------------------
st.title("📄 AI Resume Parser")

st.write(
    "Upload your resume and get AI-powered resume analysis."
)

# -----------------------------------
# File Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# -----------------------------------
# Resume Processing
# -----------------------------------
if uploaded_file is not None:

    text = ""

    try:

        # Read PDF
        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                extracted_text = page.extract_text()

                if extracted_text:
                    text += extracted_text

        # -----------------------------------
        # Extracted Text
        # -----------------------------------
        st.subheader("📑 Extracted Resume Text")

        st.text_area(
            "Resume Content",
            text[:3000],
            height=250
        )

        # -----------------------------------
        # Clean Resume
        # -----------------------------------
        cleaned_resume = clean_resume(text)

        # -----------------------------------
        # TF-IDF Transformation
        # -----------------------------------
        resume_vector = tfidf.transform([cleaned_resume])

        # -----------------------------------
        # Prediction
        # -----------------------------------
        prediction = model.predict(resume_vector)

        st.subheader("🎯 Predicted Category")

        st.success(prediction[0])

        # -----------------------------------
        # Skill Extraction
        # -----------------------------------
        found_skills = []

        for skill in skills:

            if skill in cleaned_resume:
                found_skills.append(skill)

        st.subheader("🛠 Extracted Skills")

        if found_skills:

            st.write(found_skills)

        else:

            st.warning("No major technical skills detected.")

        # -----------------------------------
        # Resume Score
        # -----------------------------------
        score = min(len(found_skills) * 5, 100)

        st.subheader("📊 Resume Score")

        st.progress(score)

        st.write(f"Resume Score: {score}/100")

        # -----------------------------------
        # Score Status
        # -----------------------------------
        if score < 50:

            st.error("Low Resume Score")

        elif score < 80:

            st.warning("Average Resume Score")

        else:

            st.success("Excellent Resume Score")

        # -----------------------------------
        # Recommendations
        # -----------------------------------
        st.subheader("💡 Recommendations")

        recommendations = []

        if "aws" not in found_skills:
            recommendations.append("Learn AWS or Cloud Computing")

        if "docker" not in found_skills:
            recommendations.append("Add Docker projects")

        if "machine learning" not in found_skills:
            recommendations.append("Explore Machine Learning basics")

        if "github" not in found_skills:
            recommendations.append("Add GitHub project links")

        if score < 50:
            recommendations.append(
                "Add more technical projects and certifications"
            )

        if recommendations:

            for rec in recommendations:
                st.info(rec)

        else:

            st.success(
                "Your resume looks strong for technical roles."
            )

    except Exception as e:

        st.error(f"Error processing resume: {e}")