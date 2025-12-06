import mlflow
import random

mlflow.set_tracking_uri("http://ec2-35-172-150-63.compute-1.amazonaws.com:5000//")
mlflow.set_experiment("Random Number Experiment")   

with mlflow.start_run():
    random_number = random.randint(1, 100)
    mlflow.log_param("random_seed", 42)
    mlflow.log_metric("random_number", random_number)   

    mlflow.log_metric("squared_random_number", random_number ** 2)
    mlflow.log_metric("cubed_random_number", random_number ** 3)

    print(f"Logged random number: {random_number}, its square and cube to MLflow.")
