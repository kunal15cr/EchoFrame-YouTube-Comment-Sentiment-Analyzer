import os
import mlflow
import pytest
import pandas as pd
import pickle
import logging
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# MLflow configuration
# -------------------------------
mlflow.set_tracking_uri("http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/")

# -------------------------------
# Logging configuration
# -------------------------------
logger = logging.getLogger("test_model_performance")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -------------------------------
# Parametrized test
# -------------------------------
@pytest.mark.parametrize(
    "model_name, stage, holdout_path, thresholds",
    [
        (
            "my_model",
            "Staging",
            "data/interim/test_processed.csv",
            {"accuracy": 0.40, "precision": 0.40, "recall": 0.40, "f1": 0.40},
        ),
    ],
)
def test_model_meets_performance_thresholds(model_name, stage, holdout_path, thresholds):
    client = MlflowClient()

    # -------------------------------
    # 1. Get latest model version
    # -------------------------------
    try:
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        assert latest_versions, f"No model found in stage '{stage}'"
        latest_version = latest_versions[0]
        run_id = latest_version.run_id
        logger.info(f"Loaded model '{model_name}' version {latest_version.version} from stage '{stage}'")
    except Exception as e:
        pytest.fail(f"Failed to fetch latest model from MLflow Registry: {e}")

    # -------------------------------
    # 2. Load model
    # -------------------------------
    try:
        model_uri = f"runs:/{run_id}/lgbm_model"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        pytest.fail(f"Failed to load sklearn model {model_uri}: {e}")

    # -------------------------------
    # 3. Load TF-IDF vectorizer
    # -------------------------------
    try:
        vec_path = client.download_artifacts(run_id, "tfidf_vectorizer.pkl")
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        pytest.fail(f"Failed to load TF-IDF vectorizer: {e}")

    # -------------------------------
    # 4. Load test data
    # -------------------------------
    if not os.path.exists(holdout_path):
        pytest.skip(f"Holdout data not found: {holdout_path}")
    df = pd.read_csv(holdout_path)

    required_cols = ["clean_comment", "category"]
    for col in required_cols:
        if col not in df.columns:
            pytest.fail(f"Required column '{col}' missing from holdout data")

    X_raw = df["clean_comment"].astype(str).tolist()
    y_true = df["category"]

    # -------------------------------
    # 5. Transform text
    # -------------------------------
    try:
        X_test_tfidf = vectorizer.transform(X_raw)
    except Exception as e:
        pytest.fail(f"Vectorizer.transform failed: {e}")

    # -------------------------------
    # 6. Predict
    # -------------------------------
    try:
        y_pred = model.predict(X_test_tfidf)
    except Exception as e:
        pytest.fail(f"Model prediction failed: {e}")

    # -------------------------------
    # 7. Evaluate
    # -------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

    logger.info(
        f"Model performance - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}"
    )

    # -------------------------------
    # 8. Assert thresholds
    # -------------------------------
    assert acc >= thresholds["accuracy"]
    assert prec >= thresholds["precision"]
    assert rec >= thresholds["recall"]
    assert f1 >= thresholds["f1"]
