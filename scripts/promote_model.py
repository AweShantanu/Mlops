# promote_model.py
import mlflow
from src.mlflow_config import setup_mlflow


def promote_model():
    """
    Promote model using TAGS (not aliases, not stages)
    Logic:
    env=staging  -->  env=production
    """

    # ✅ Centralized MLflow / DagsHub setup
    setup_mlflow()

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # ---------------------------------------------------
    # 1️⃣ Find model version with tag env=staging
    # ---------------------------------------------------
    staging_version = None

    versions = client.search_model_versions(f"name='{model_name}'")

    for v in versions:
        tags = v.tags or {}
        if tags.get("env") == "staging":
            staging_version = v.version
            break

    if not staging_version:
        raise RuntimeError(
            "❌ No model version found with tag env=staging.\n"
            "Fix: Run register_model.py first."
        )

    print(f"🚀 Found STAGING model version: {staging_version}")

    # ---------------------------------------------------
    # 2️⃣ (Optional) Clean old production tags
    # ---------------------------------------------------
    for v in versions:
        if v.tags.get("env") == "production":
            client.set_model_version_tag(
                name=model_name,
                version=v.version,
                key="env",
                value="archived"
            )
            print(f"📦 Archived old production version {v.version}")

    # ---------------------------------------------------
    # 3️⃣ Promote staging → production
    # ---------------------------------------------------
    client.set_model_version_tag(
        name=model_name,
        version=staging_version,
        key="env",
        value="production"
    )

    print(f"✅ Model version {staging_version} promoted to PRODUCTION")


if __name__ == "__main__":
    promote_model()
