import os
import unittest

from flask_app.app import app


class TestFlaskApp(unittest.TestCase):
    """
    Integration tests for Flask app with real MLflow model + scaler.
    """

    @classmethod
    def setUpClass(cls):
        # Ensure DAGSHUB token exists
        if not os.getenv("DAGSHUB_TOKEN_HEALTH"):
            raise EnvironmentError(
                "DAGSHUB_TOKEN_HEALTH environment variable is required for tests"
            )

        app.testing = True
        cls.client = app.test_client()

    # -----------------------
    # Home page
    # -----------------------
    def test_home_page(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Insurance Cost Prediction", response.data)
        self.assertIn(b"Predict Insurance Cost", response.data)

    # -----------------------
    # Prediction endpoint
    # -----------------------
    def test_prediction_endpoint(self):
        payload = {
            "age": "35",
            "gender": "male",
            "bmi": "24.5",
            "bloodpressure": "120",
            "diabetic": "no",
            "children": "1",
            "smoker": "no",
        }

        response = self.client.post("/predict", data=payload)

        self.assertEqual(response.status_code, 200)

        # Ensure prediction rendered
        self.assertIn(b"Predicted Insurance Cost", response.data)

        # Ensure error not shown
        self.assertNotIn(b"Error:", response.data)

    # -----------------------
    # Metrics endpoint
    # -----------------------
    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)

        # Validate actual metric names defined in app.py
        self.assertIn(b"app_request_total", response.data)
        self.assertIn(b"app_request_latency_seconds", response.data)
        self.assertIn(b"model_prediction_total", response.data)
        self.assertIn(b"model_prediction_latency_seconds", response.data)
        self.assertIn(b"model_prediction_value", response.data)


if __name__ == "__main__":
    unittest.main()
