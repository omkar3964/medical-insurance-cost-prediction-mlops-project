import unittest
from unittest.mock import patch
from flask_app.app import app


class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    # 1 Home Route Test
    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Insurance", response.data)

    # 2 Prediction Route Test (MOCKED)
    @patch("flask_app.app.load_artifacts")
    @patch("flask_app.app.model")
    @patch("flask_app.app.scaler")
    def test_prediction_endpoint(self, mock_scaler, mock_model, mock_load_artifacts):
        # Mock scaler
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]

        # Mock model prediction
        mock_model.predict.return_value = [12345.67]

        payload = {
            "age": "30",
            "gender": "male",
            "bmi": "24.5",
            "bloodpressure": "120",
            "diabetic": "no",
            "children": "1",
            "smoker": "no",
        }

        response = self.client.post("/predict", data=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Result", response.data)

    # 3 Metrics Endpoint Test
    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"app_request_total", response.data)
        self.assertIn(b"model_prediction_total", response.data)


if __name__ == "__main__":
    unittest.main()
