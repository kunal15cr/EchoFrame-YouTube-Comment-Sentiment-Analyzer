# predict.py
import os
import re
from functools import lru_cache
from typing import List, Tuple, Dict, Any
import io
import sys
import subprocess
import yaml

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import matplotlib.dates as mdates

# ---------------------------
# Config (env overrides allowed)
# ---------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-3-229-124-18.compute-1.amazonaws.com:5000/"
)
MODEL_NAME = os.getenv("MODEL_NAME", "my_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "./tfidf_vectorizer.pkl")

# Pre-compile regex patterns
NEWLINE_PATTERN = re.compile(r"\n")
NON_ALNUM_PATTERN = re.compile(r"[^A-Za-z0-9\s!?.,]")

# Initialize nltk objects (assumes corpora are installed)
_LEMMATIZER = WordNetLemmatizer()
try:
    _STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
except Exception:
    # fallback to a small set if nltk data not available
    _STOP_WORDS = {"the", "a", "and", "is", "in", "it"} - {"not", "but", "however", "no", "yet"}


# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_comment(comment: str) -> str:
    try:
        text = comment.lower().strip()
        text = NEWLINE_PATTERN.sub(" ", text)
        text = NON_ALNUM_PATTERN.sub("", text)

        tokens = [w for w in text.split() if w not in _STOP_WORDS]
        lemmatized_tokens = [_LEMMATIZER.lemmatize(w) for w in tokens]
        return " ".join(lemmatized_tokens)
    except Exception:
        # return original text as fallback
        return comment


# ---------------------------
# Model + Vectorizer Loading
# ---------------------------
@lru_cache(maxsize=1)
def get_model_and_vectorizer() -> Tuple[Any, Any]:
    """
    Load and cache model + vectorizer. Attempts to fetch model dependencies (non-fatal).
    Raises RuntimeError upon failure.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _ = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

        # try to fetch dependencies and pip install them (best-effort)
        try:
            deps_path = mlflow.pyfunc.get_model_dependencies(model_uri)
            if deps_path:
                with open(deps_path, "r", encoding="utf-8") as f:
                    env_yaml = yaml.safe_load(f)
                pip_deps = []
                for dep in env_yaml.get("dependencies", []):
                    if isinstance(dep, dict) and "pip" in dep:
                        pip_deps = dep.get("pip", [])
                        break
                if pip_deps:
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "install", *pip_deps], check=False)
                    except Exception:
                        pass
        except Exception:
            # not fatal
            pass

        model = mlflow.pyfunc.load_model(model_uri)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as exc:
        raise RuntimeError(f"Failed to load model/vectorizer: {exc}") from exc


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
    """
    Accepts a list of dicts with keys: 'text' and 'timestamp' (timestamp optional).
    Returns a list of dicts like: {"comment": text, "sentiment": "<pred>", "timestamp": "<orig>"}
    This mirrors the Flask output (Option A).
    """
    if not comments_with_ts:
        raise ValueError("No comments provided for prediction_with_timestamps.")

    comments = []
    timestamps = []
    for item in comments_with_ts:
        if not isinstance(item, dict) or "text" not in item:
            raise ValueError("Each item must be a dict containing at least the key 'text'.")
        comments.append(item["text"])
        timestamps.append(item.get("timestamp"))

    preds = predict_comments(comments)
    # convert predictions to string to match your Flask behavior
    preds_str = [str(p) for p in preds]

    response = [
        {"comment": c, "sentiment": s, "timestamp": t}
        for c, s, t in zip(comments, preds_str, timestamps)
    ]
    return response


# ---------------------------
# Chart / Wordcloud helpers
# ---------------------------
def generate_pie_chart_bytes(sentiment_counts: dict) -> bytes:
    """
    sentiment_counts: dict with keys '1','0','-1' or ints
    returns PNG bytes
    """
    # Prepare sizes
    pos = int(sentiment_counts.get("1", sentiment_counts.get(1, 0)))
    neu = int(sentiment_counts.get("0", sentiment_counts.get(0, 0)))
    neg = int(sentiment_counts.get("-1", sentiment_counts.get(-1, 0)))
    sizes = [pos, neu, neg]

    if sum(sizes) == 0:
        raise ValueError("Sentiment counts sum to zero.")

    labels = ["Positive", "Neutral", "Negative"]
    # Use explicit colors to mimic original; matplotlib defaults otherwise
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
    if not comments:
        raise ValueError("No comments provided for wordcloud.")

    preprocessed = [preprocess_comment(c) for c in comments]
    text = " ".join(preprocessed)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Blues",
        stopwords=set(stopwords.words("english")),
        collocations=False
    ).generate(text)

    img = wc.to_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def generate_trend_graph_bytes(sentiment_data: List[dict]) -> bytes:
    """
    sentiment_data: list of {"comment":..., "sentiment": int, "timestamp": iso-string}
    returns PNG bytes for monthly sentiment % trend
    """
    if not sentiment_data:
        raise ValueError("No sentiment data provided.")

    df = pd.DataFrame(sentiment_data)
    if "timestamp" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("sentiment_data must include 'timestamp' and 'sentiment' fields.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['sentiment'] = df['sentiment'].astype(int)

    monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
    monthly_totals = monthly_counts.sum(axis=1).replace(0, 1)  # avoid divide by zero
    monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

    # ensure columns -1,0,1 present
    for col in [-1, 0, 1]:
        if col not in monthly_percentages.columns:
            monthly_percentages[col] = 0
    monthly_percentages = monthly_percentages[[-1, 0, 1]]

    # plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = { -1: 'red', 0: 'gray', 1: 'green' }
    labels = { -1: 'Negative', 0: 'Neutral', 1: 'Positive' }

    for sentiment_value in [-1, 0, 1]:
        ax.plot(
            monthly_percentages.index,
            monthly_percentages[sentiment_value],
            marker='o',
            linestyle='-',
            label=labels[sentiment_value],
            color=colors[sentiment_value]
        )

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
