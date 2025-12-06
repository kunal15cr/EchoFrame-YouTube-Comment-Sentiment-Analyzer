import mlflow
import pytest
from mlflow.tracking import MlflowClient

# remote tracking server used in this repo
mlflow.set_tracking_uri("http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/")


def _pick_latest_version(client: MlflowClient, model_name: str) -> str | None:
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        return None
    return str(max(versions, key=lambda v: int(v.version)).version)


@pytest.mark.parametrize("model_name", ["my_model"])
def test_model_signature_exists(model_name):
    """Single check: ensure registered model has a saved signature (inputs/outputs)."""
    client = MlflowClient()
    ver = _pick_latest_version(client, model_name)
    assert ver is not None, f"No versions found for model '{model_name}'"

    model_uri = f"models:/{model_name}/{ver}"
    try:
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    except ModuleNotFoundError as e:
        pytest.skip(f"Missing runtime dependency while loading model {model_uri}: {e}")
    except Exception as e:
        pytest.fail(f"Failed to load model {model_uri}: {e}")

    # extract signature from model metadata
    meta = getattr(pyfunc_model, "metadata", None) or getattr(pyfunc_model, "_model_meta", None)
    signature = getattr(meta, "signature", None) if meta is not None else None

    assert signature is not None, f"Model {model_name} v{ver} is missing a signature (inputs/outputs)"
    assert getattr(signature, "inputs", None) is not None, "Signature has no inputs"
    assert getattr(signature, "outputs", None) is not None, "Signature has no outputs"