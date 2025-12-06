import os
import pickle
import mlflow
import pytest
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# remote tracking server used in this repo
mlflow.set_tracking_uri("http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/")


@pytest.mark.parametrize(
    "model_name, stage, holdout_path, vectorizer_path, thresholds",
    [
        ("my_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl",
         {"accuracy": 0.40, "precision": 0.40, "recall": 0.40, "f1": 0.40}),
    ],
)
def test_model_meets_performance_thresholds(model_name, stage, holdout_path, vectorizer_path, thresholds):
    client = MlflowClient()

    latest = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest[0].version if latest else None
    assert latest_version is not None, f"No model in stage '{stage}' for '{model_name}'"

    model_uri = f"models:/{model_name}/{latest_version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except ModuleNotFoundError as e:
        pytest.skip(f"Missing runtime dependency loading model {model_uri}: {e}")
    except Exception as e:
        pytest.fail(f"Failed to load model {model_uri}: {e}")

    if not os.path.exists(vectorizer_path):
        pytest.skip(f"Vectorizer file not found at '{vectorizer_path}'")

    try:
        with open(vectorizer_path, "rb") as vf:
            vectorizer = pickle.load(vf)
    except Exception as e:
        pytest.fail(f"Failed to load vectorizer '{vectorizer_path}': {e}")

    if not os.path.exists(holdout_path):
        pytest.skip(f"Holdout data not found at '{holdout_path}'")

    df = pd.read_csv(holdout_path)

    # heuristics to find text column and label column
    text_col = None
    label_col = None
    for c in ("clean_comment", "comment", "text"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # fallback to first non-numeric column
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    # assume label is column named 'category' or last column
    if "category" in df.columns:
        label_col = "category"
    else:
        label_col = df.columns[-1]

    assert text_col is not None, "Could not determine text column in holdout data"
    X_raw = df[text_col].fillna("").astype(str)
    y_true = df[label_col]

    try:
        X_tfidf = vectorizer.transform(X_raw)
    except Exception as e:
        pytest.fail(f"Vectorizer.transform failed: {e}")

    # prepare dataframe expected by model if needed
    try:
        cols = vectorizer.get_feature_names_out()
        X_df = pd.DataFrame(X_tfidf.toarray(), columns=cols)
    except Exception:
        X_df = pd.DataFrame(X_tfidf.toarray())

    try:
        y_pred = model.predict(X_df)
    except Exception as e:
        pytest.fail(f"Model.predict failed: {e}")

    # compute metrics (weighted to handle multi-class)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

    assert acc >= thresholds["accuracy"], f"accuracy {acc:.3f} < expected {thresholds['accuracy']:.3f}"
    assert prec >= thresholds["precision"], f"precision {prec:.3f} < expected {thresholds['precision']:.3f}"
    assert rec >= thresholds["recall"], f"recall {rec:.3f} < expected {thresholds['recall']:.3f}"
    assert f1 >= thresholds["f1"], f"f1 {f1:.3f} < expected {thresholds['f1']:.3f}"
# filepath: tests/test_model_performance_thresholds.py
import os
import pickle
import mlflow
import pytest
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# remote tracking server used in this repo
mlflow.set_tracking_uri("http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/")


@pytest.mark.parametrize(
    "model_name, stage, holdout_path, vectorizer_path, thresholds",
    [
        ("my_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl",
         {"accuracy": 0.40, "precision": 0.40, "recall": 0.40, "f1": 0.40}),
    ],
)
def test_model_meets_performance_thresholds(model_name, stage, holdout_path, vectorizer_path, thresholds):
    client = MlflowClient()

    latest = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest[0].version if latest else None
    assert latest_version is not None, f"No model in stage '{stage}' for '{model_name}'"

    model_uri = f"models:/{model_name}/{latest_version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except ModuleNotFoundError as e:
        pytest.skip(f"Missing runtime dependency loading model {model_uri}: {e}")
    except Exception as e:
        pytest.fail(f"Failed to load model {model_uri}: {e}")

    if not os.path.exists(vectorizer_path):
        pytest.skip(f"Vectorizer file not found at '{vectorizer_path}'")

    try:
        with open(vectorizer_path, "rb") as vf:
            vectorizer = pickle.load(vf)
    except Exception as e:
        pytest.fail(f"Failed to load vectorizer '{vectorizer_path}': {e}")

    if not os.path.exists(holdout_path):
        pytest.skip(f"Holdout data not found at '{holdout_path}'")

    df = pd.read_csv(holdout_path)

    # heuristics to find text column and label column
    text_col = None
    label_col = None
    for c in ("clean_comment", "comment", "text"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # fallback to first non-numeric column
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    # assume label is column named 'category' or last column
    if "category" in df.columns:
        label_col = "category"
    else:
        label_col = df.columns[-1]

    assert text_col is not None, "Could not determine text column in holdout data"
    X_raw = df[text_col].fillna("").astype(str)
    y_true = df[label_col]

    try:
        X_tfidf = vectorizer.transform(X_raw)
    except Exception as e:
        pytest.fail(f"Vectorizer.transform failed: {e}")

    # prepare dataframe expected by model if needed
    try:
        cols = vectorizer.get_feature_names_out()
        X_df = pd.DataFrame(X_tfidf.toarray(), columns=cols)
    except Exception:
        X_df = pd.DataFrame(X_tfidf.toarray())

    try:
        y_pred = model.predict(X_df)
    except Exception as e:
        pytest.fail(f"Model.predict failed: {e}")

    # compute metrics (weighted to handle multi-class)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

    assert acc >= thresholds["accuracy"], f"accuracy {acc:.3f} < expected {thresholds['accuracy']:.3f}"
    assert prec >= thresholds["precision"], f"precision {prec:.3f} < expected {thresholds['precision']:.3f}"
    assert rec >= thresholds["recall"], f"recall {rec:.3f} < expected {thresholds['recall']:.3f}"
    assert f1 >= thresholds["f1"], f"f1 {f1:.3f} < expected {thresholds['f1']:.3f}"