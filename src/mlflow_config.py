import mlflow
import dagshub
import os

def setup_mlflow():
    # Get DagsHub token from environment variable
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    if dagshub_token:
        # Set authentication for MLflow
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        print("✅ DagsHub authentication configured")
    else:
        print("⚠️  Warning: DAGSHUB_TOKEN not found in environment variables")
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
    
    # Set tracking URI
    mlflow.set_tracking_uri("https://dagshub.com/AweShantanu/Mlops.mlflow")
    
    # Initialize DagsHub
    dagshub.init(
        repo_owner="AweShantanu",
        repo_name="Mlops",
        mlflow=True
    )