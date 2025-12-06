import mlflow
import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# Set your remote MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name, stage", [
    ("my_model", "staging"),   # ← Updated to your actual model name
])
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()

    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        model_uri = f"models:/{model_name}/{latest_version}"
        print(f"Trying to load: {model_uri}")

        # Load the MLflow model
        model = mlflow.pyfunc.load_model(model_uri)

        assert model is not None, "Model failed to load"

        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
