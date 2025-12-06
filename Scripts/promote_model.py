import os
import mlflow
import logging

# ---------------------------
# Logging configuration
# ---------------------------
logger = logging.getLogger("model_promotion")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ---------------------------
# MLflow configuration
# ---------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-35-172-150-63.compute-1.amazonaws.com:5000/"
)
MODEL_NAME = os.getenv("MODEL_NAME", "my_model")
MODEL_TAG_FILTER = os.getenv("MODEL_TAG_FILTER", None)  # Optional: "stage=Staging"

def promote_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # Get all versions of the model
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not all_versions:
        logger.error(f"No versions found for model '{MODEL_NAME}'")
        return

    # Optional: filter by tags
    if MODEL_TAG_FILTER:
        key, value = MODEL_TAG_FILTER.split("=")
        filtered_versions = [
            v for v in all_versions
            if v.tags.get(key) == value
        ]
    else:
        filtered_versions = all_versions

    if not filtered_versions:
        logger.error(f"No model versions match the tag filter '{MODEL_TAG_FILTER}'")
        return

    # Find latest staging version by creation timestamp
    staging_versions = [v for v in filtered_versions if v.current_stage == "Staging"]
    if not staging_versions:
        logger.error(f"No model found in 'Staging' stage for {MODEL_NAME}")
        return

    latest_staging = max(staging_versions, key=lambda x: x.creation_timestamp)

    # Archive current production versions
    prod_versions = [v for v in filtered_versions if v.current_stage == "Production"]
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived"
        )
        logger.info(f"Archived production model version {v.version}")

    # Promote latest staging version to production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_staging.version,
        stage="Production"
    )
    logger.info(f"Promoted model version {latest_staging.version} to Production")
    print(f"Model version {latest_staging.version} promoted to Production")


if __name__ == "__main__":
    promote_model()
