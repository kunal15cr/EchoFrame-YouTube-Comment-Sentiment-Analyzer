import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns

# logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('model_evaluation')


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data and handle missing values."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load ML model from pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model performance."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug('Model evaluation completed')
        return report, cm

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Save and log the confusion matrix."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
    except Exception as e:
        logger.error("Error logging confusion matrix: %s", e)
        raise


def main():
    mlflow.set_tracking_uri("http://ec2-3-235-148-10.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run():
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

            # Load parameters
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model + vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Log model parameters
            if hasattr(model, 'get_params'):
                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Log artifacts
            mlflow.sklearn.log_model(model, "lgbm_model")
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Evaluate
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification report
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics.get('precision', 0),
                        f"test_{label}_recall": metrics.get('recall', 0),
                        f"test_{label}_f1": metrics.get('f1-score', 0)
                    })

            log_confusion_matrix(cm, "Test_Data")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error("Failed to complete model evaluation: %s", e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
