# tests/test_model.py

import unittest
import mlflow
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Centralized MLflow + DagsHub auth
        from src.mlflow_config import setup_mlflow
        setup_mlflow()

        cls.model_name = "my_model"
        cls.model_version = cls.get_model_version_by_tag(
            cls.model_name,
            tag_key="env",
            tag_value="staging"
        )

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        cls.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    @staticmethod
    def get_model_version_by_tag(model_name, tag_key, tag_value):
        """
        Fetch model version using model version tags.
        Works reliably with DagsHub MLflow.
        """
        client = mlflow.MlflowClient()

        versions = client.search_model_versions(
            filter_string=f"name='{model_name}' and tags.{tag_key}='{tag_value}'",
            order_by=["version_number DESC"]
        )

        if not versions:
            raise RuntimeError(
                f"No model version found with tag {tag_key}={tag_value}"
            )

        return versions[0].version

    # ---------------------- Tests ----------------------

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        text = "hi how are you"
        features = self.vectorizer.transform([text])

        input_df = pd.DataFrame(
            features.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

        preds = self.model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(preds), input_df.shape[0])

    def test_model_performance(self):
        X = self.holdout_data.iloc[:, :-1]
        y = self.holdout_data.iloc[:, -1]

        y_pred = self.model.predict(X)

        self.assertGreaterEqual(accuracy_score(y, y_pred), 0.40)
        self.assertGreaterEqual(precision_score(y, y_pred), 0.40)
        self.assertGreaterEqual(recall_score(y, y_pred), 0.40)
        self.assertGreaterEqual(f1_score(y, y_pred), 0.40)


if __name__ == "__main__":
    unittest.main()
