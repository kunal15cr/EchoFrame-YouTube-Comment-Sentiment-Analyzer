# predict.py
import os
import re
from functools import lru_cache
from typing import List, Tuple, Dict, Any
import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates

# ---------------------------
# Config (env overrides allowed)
# ---------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/"
)
MODEL_NAME = os.getenv("MODEL_NAME", "my_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# Pre-compile regex patterns
NEWLINE_PATTERN = re.compile(r"\n")
NON_ALNUM_PATTERN = re.compile(r"[^A-Za-z0-9\s!?.,]")

# Initialize nltk objects
_LEMMATIZER = WordNetLemmatizer()
try:
    _STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
except Exception:
    _STOP_WORDS = {"the", "a", "and", "is", "in", "it"} - {"not", "but", "however", "no", "yet"}

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_comment(comment: str) -> str:
    text = comment.lower().strip()
    text = NEWLINE_PATTERN.sub(" ", text)
    text = NON_ALNUM_PATTERN.sub("", text)
    tokens = [w for w in text.split() if w not in _STOP_WORDS]
    lemmatized_tokens = [_LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(lemmatized_tokens)

# ---------------------------
# Model + Vectorizer Loading from MLflow
# ---------------------------
@lru_cache(maxsize=1)
def get_model_and_vectorizer() -> Tuple[Any, Any]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load vectorizer artifact from MLflow
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_version_info = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
    run_id = model_version_info.run_id
    vectorizer_path = client.download_artifacts(run_id, "tfidf_vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

# ---------------------------
# Prediction functions
# ---------------------------
def predict_comments(comments: List[str]) -> List[float]:
    if not comments:
        raise ValueError("No comments provided for prediction.")
    model, vectorizer = get_model_and_vectorizer()
    preprocessed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(preprocessed)
    raw_preds = model.predict(transformed)
    return [float(p) for p in raw_preds]

def predict_with_timestamps(comments_with_ts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    comments = []
    timestamps = []
    for item in comments_with_ts:
        if not isinstance(item, dict) or "text" not in item:
            raise ValueError("Each item must be a dict with 'text' key.")
        comments.append(item["text"])
        timestamps.append(item.get("timestamp"))
    preds = predict_comments(comments)
    preds_str = [str(p) for p in preds]
    response = [{"comment": c, "sentiment": s, "timestamp": t} for c, s, t in zip(comments, preds_str, timestamps)]
    return response

# ---------------------------
# Chart / Wordcloud helpers
# ---------------------------
def generate_pie_chart_bytes(sentiment_counts: dict) -> bytes:
    pos = int(sentiment_counts.get("1", sentiment_counts.get(1, 0)))
    neu = int(sentiment_counts.get("0", sentiment_counts.get(0, 0)))
    neg = int(sentiment_counts.get("-1", sentiment_counts.get(-1, 0)))
    sizes = [pos, neu, neg]
    if sum(sizes) == 0:
        raise ValueError("Sentiment counts sum to zero.")
    labels = ["Positive", "Neutral", "Negative"]
    colors = ['#36A2EB', '#C9CBCF', '#FF6384']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, textprops={'color': 'w'})
    ax.axis("equal")
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def generate_wordcloud_bytes(comments: List[str]) -> bytes:
    preprocessed = [preprocess_comment(c) for c in comments]
    text = " ".join(preprocessed)
    wc = WordCloud(width=800, height=400, background_color="black", colormap="Blues", stopwords=set(stopwords.words("english")), collocations=False).generate(text)
    img = wc.to_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

def generate_trend_graph_bytes(sentiment_data: List[dict]) -> bytes:
    df = pd.DataFrame(sentiment_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['sentiment'] = df['sentiment'].astype(int)
    monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
    monthly_totals = monthly_counts.sum(axis=1).replace(0, 1)
    monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
    for col in [-1, 0, 1]:
        if col not in monthly_percentages.columns:
            monthly_percentages[col] = 0
    monthly_percentages = monthly_percentages[[-1, 0, 1]]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = { -1: 'red', 0: 'gray', 1: 'green' }
    labels = { -1: 'Negative', 0: 'Neutral', 1: 'Positive' }
    for sentiment_value in [-1, 0, 1]:
        ax.plot(monthly_percentages.index, monthly_percentages[sentiment_value], marker='o', linestyle='-', label=labels[sentiment_value], color=colors[sentiment_value])
    ax.set_title("Monthly Sentiment Percentage Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Percentage of Comments (%)")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
