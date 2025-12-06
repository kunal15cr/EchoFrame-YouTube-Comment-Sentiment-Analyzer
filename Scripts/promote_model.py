# src/utils/promote_model.py

import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------
# Logging Configuration
# ---------------------------
logger = logging.getLogger("model_promotion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_promotion_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------------------
# Model Promotion Function
# ---------------------------
def promote_model(model_name: str = "my_model", tracking_uri: str = None):
    """
    Promote the latest staging model to Production in MLflow Model Registry.
    Archives existing production versions.
    """
    try:
        tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/"
        )
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Get the latest staging version
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            logger.error("No model found in 'Staging' stage for %s", model_name)
            return

        latest_staging_version = staging_versions[0].version
        logger.info("Latest staging version: %s", latest_staging_version)

        # Archive current production versions
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for v in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
            logger.info("Archived production version: %s", v.version)

        # Promote staging to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging_version,
            stage="Production"
        )
        logger.info("Model version %s promoted to Production", latest_staging_version)
        print(f"Model version {latest_staging_version} promoted to Production")

    except Exception as e:
        logger.error("Failed to promote model %s: %s", model_name, e)
        print(f"Error promoting model: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    promote_model(model_name="yt_chrome_plugin_model")
