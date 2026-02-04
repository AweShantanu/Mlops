import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import tempfile
from src.logger import logging



def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    from src.mlflow_config import setup_mlflow
    setup_mlflow()
    mlflow.set_experiment("my-dvc-pipeline")
    
    with mlflow.start_run() as run:
        try:
            # Load model and data
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            
            # Evaluate model
            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # WORKAROUND: Save model to temp directory and log as artifacts
            print("Saving model artifacts...")
            with tempfile.TemporaryDirectory() as tmpdir:
                model_dir = os.path.join(tmpdir, "model")
                
                # Save the model using mlflow.sklearn.save_model
                mlflow.sklearn.save_model(
                    sk_model=clf,
                    path=model_dir
                )
                
                logging.info(f"Model saved to temporary directory: {model_dir}")
                print(f"✅ Model saved locally to: {model_dir}")
                
                # List files in the model directory
                print(f"Model directory contents:")
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        print(f"  - {filepath}")
                
                # Log the entire model directory as artifacts
                print("Uploading model artifacts to MLflow...")
                mlflow.log_artifacts(model_dir, artifact_path="model")
                
                logging.info("Model artifacts logged to MLflow")
                print("✅ Model artifacts uploaded")
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            
            # Wait for artifacts to sync
            import time
            print("Waiting for artifacts to sync...")
            time.sleep(3)
            
            # Verify artifacts
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"\nArtifacts in MLflow: {[a.path for a in artifacts]}")
            
            # Save model info for later registration
            save_model_info(
                run.info.run_id,
                "model",
                "reports/experiment_info.json"
            )
            
            logging.info(f"Model evaluation completed successfully. Run ID: {run.info.run_id}")
            print(f"\n✅ Success! Run ID: {run.info.run_id}")
            print(f"View run at: https://dagshub.com/AweShantanu/Mlops.mlflow/#/experiments/3/runs/{run.info.run_id}")
            
        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    main()