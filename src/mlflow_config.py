import os
import mlflow
import dagshub
from src.logger import logging


def setup_mlflow():
    """
    CI/Production-safe MLflow + Dagshub setup.
    Uses token-based auth only (no OAuth).
    """

    # Prefer CAPSTONE_TEST (prod), fallback to DAGSHUB_TOKEN (local)
    token = os.getenv("CAPSTONE_TEST") or os.getenv("DAGSHUB_TOKEN")

    if not token:
        raise EnvironmentError(
            "Neither CAPSTONE_TEST nor DAGSHUB_TOKEN is set"
        )

    # 🔐 Token-based auth (NO OAuth)
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # Repo config
    dagshub_url = "https://dagshub.com"
    repo_owner = "AweShantanu"
    repo_name = "Mlops"

    mlflow.set_tracking_uri(
        f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    )

    # Prevent OAuth by explicitly adding token
    dagshub.auth.add_app_token(token)

    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True
    )

    logging.info("✅ MLflow + Dagshub initialized using token auth")
