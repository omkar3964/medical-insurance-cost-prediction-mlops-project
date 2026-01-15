import unittest
import mlflow
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TestInsuranceCostModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load model, scaler, and holdout data once for all tests
        """

        # DagsHub / MLflow Auth
        dagshub_token = os.getenv("DAGSHUB_TOKEN_HEALTH")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN_HEALTH environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri("https://dagshub.com/omalbhare/medical-insurance-cost-prediction-mlops-project.mlflow")

        # Load Production Model
        cls.model_name = "InsuranceCostModel"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load Scaler
        cls.scaler = joblib.load("models/scaler.pkl")

        # Load Holdout Test Data
        cls.test_data = pd.read_csv("data/processed/test.csv")

        cls.numeric_cols = ["age", "bmi", "bloodpressure"]
        cls.final_columns = ["age","gender","bmi","bloodpressure","diabetic","children","smoker",]

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions(model_name, stages=[stage])
        return latest[0].version if latest else None

    # 1️ Load Test
    def test_model_loaded_successfully(self):
        self.assertIsNotNone(self.model, "Model failed to load from MLflow registry")

    # 2️ Signature / Schema Test (WITH SCALER)
    def test_model_input_output_signature(self):
        raw_input = pd.DataFrame([[25, 0, 22.5, 120, 1, 0, 0]],  columns=self.final_columns, )

        # Apply scaler to numeric columns
        scaled_numeric = self.scaler.transform(raw_input[self.numeric_cols])

        processed_input = pd.DataFrame(
            [[
                scaled_numeric[0, 0],
                raw_input["gender"].iloc[0],
                scaled_numeric[0, 1],
                scaled_numeric[0, 2],
                raw_input["diabetic"].iloc[0],
                raw_input["children"].iloc[0],
                raw_input["smoker"].iloc[0],
            ]],
            columns=self.final_columns,
        )

        prediction = self.model.predict(processed_input)

        self.assertEqual(processed_input.shape, (1, 7))
        self.assertEqual(len(prediction), 1)
        self.assertIsInstance(prediction[0], (float, np.floating))

    # 3️ Performance Test (WITH SCALER)
    def test_model_performance_thresholds(self):
        X_raw = self.test_data.drop(columns=["claim"])
        y_test = self.test_data["claim"]

        # Apply same preprocessing as inference
        scaled_numeric = self.scaler.transform(X_raw[self.numeric_cols])

        X_processed = pd.DataFrame(
            np.column_stack([
                scaled_numeric[:, 0],
                X_raw["gender"],
                scaled_numeric[:, 1],
                scaled_numeric[:, 2],
                X_raw["diabetic"],
                X_raw["children"],
                X_raw["smoker"],
            ]),
            columns=self.final_columns,
        )

        y_pred = self.model.predict(X_processed)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Quality Gates
        MAX_RMSE = 5000
        MAX_MAE = 4000
        MIN_R2 = 0.50

        self.assertLessEqual(rmse, MAX_RMSE, f"RMSE too high: {rmse}")
        self.assertLessEqual(mae, MAX_MAE, f"MAE too high: {mae}")
        self.assertGreaterEqual(r2, MIN_R2, f"R² too low: {r2}")


if __name__ == "__main__":
    unittest.main()
