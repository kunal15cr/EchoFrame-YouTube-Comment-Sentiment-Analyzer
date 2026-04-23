# 🚀 EchoFrame: YouTube Comment Sentiment Analyzer

**Tagline:** Turn noisy YouTube comments into clear sentiment insights.

EchoFrame is an ML-powered system that analyzes YouTube comments and classifies them into **positive, negative, or neutral sentiment**, with APIs for predictions and visual insights.

---

## 📌 Problem

YouTube videos receive thousands of comments, making manual analysis difficult.

- Hard to track audience sentiment  
- Feedback gets lost  
- Decisions become slow  

---

## 💡 Solution

EchoFrame processes comments and provides:

- Sentiment prediction (`-1`, `0`, `1`)  
- Charts, word clouds, and trend analysis  

---

## 🏗️ Architecture

Input → Preprocessing → TF-IDF → Model → FastAPI → Output

---

## ⚙️ Tech Stack

- Python  
- FastAPI  
- MLflow  
- NLTK  
- TF-IDF  
- Matplotlib, WordCloud  
- DVC, Docker  

---

## 📊 Data

**Input:** List of comments  

**Output:**  
- `1` → Positive  
- `0` → Neutral  
- `-1` → Negative  

---

## 🤖 Model

- NLP preprocessing + lemmatization  
- TF-IDF vectorization  
- MLflow model serving  

**Baseline:**
- Accuracy ≥ 0.40  
- Precision ≥ 0.40  
- Recall ≥ 0.40  
- F1 ≥ 0.40  

---

## 🔌 API

**Base URL:**  
http://localhost:8000

### Predict
POST /predict

```json
{
  "comments": [
    "Great video!",
    "Not useful"
  ]
}
