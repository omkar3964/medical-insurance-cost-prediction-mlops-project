import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TestInsuranceCostModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_TOKEN_HEALTH")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN_HEALTH environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri("https://dagshub.com/omalbhare/medical-insurance-cost-prediction-mlops-project.mlflow")

        cls.model_name = "InsuranceCostModel"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # ✔ Already scaled
        cls.test_data = pd.read_csv("data/processed/test.csv")

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions(model_name, stages=[stage])
        return latest[0].version

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        sample_input = self.test_data.drop(columns=["claim"]).iloc[[0]]
        prediction = self.model.predict(sample_input)

        self.assertEqual(len(prediction), 1)
        self.assertIsInstance(prediction[0], (float, np.floating))

    def test_model_performance_thresholds(self):
        X_test = self.test_data.drop(columns=["claim"])
        y_test = self.test_data["claim"]

        y_pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        self.assertLessEqual(rmse, 5200)
        self.assertLessEqual(mae, 4000)
        self.assertGreaterEqual(r2, 0.80)


if __name__ == "__main__":
    unittest.main()
