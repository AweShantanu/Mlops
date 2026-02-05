import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
import time

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        logging.info(f"Attempting to register model from: {model_uri}")
        
        # Wait a bit for DagsHub to sync artifacts
        print("Waiting for artifacts to sync with DagsHub...")
        time.sleep(5)
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        logging.info(f'Model {model_name} version {model_version.version} registered successfully.')
        print(f"✅ Model registered as version {model_version.version}")
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        #client.transition_model_version_stage(
        #    name=model_name,
        #    version=model_version.version,
        #    stage="Staging"
        #)

        #client.set_registered_model_alias(
        #    name=model_name,
        #    alias="staging",
        #    version=model_version.version
        #)
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="env",
            value="staging"
        )

        
        logging.info(f'Model {model_name} version {model_version.version} transitioned to Staging.')
        print(f"✅ Model transitioned to Staging stage")
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    from src.mlflow_config import setup_mlflow
    setup_mlflow()
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        print(f"Run ID: {model_info['run_id']}")
        print(f"Model Path: {model_info['model_path']}")
        
        model_name = "my_model"
        register_model(model_name, model_info)
        
        print("\n✅ Model registration completed successfully!")
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()