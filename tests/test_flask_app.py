import unittest
import json
from flask_app.app import app


class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create Flask test client once for all tests
        """
        app.testing = True
        cls.client = app.test_client()

    # 1️ Home Route Test
    def test_home_page_loads(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Insurance", response.data)  # Adjust keyword if needed

    # 2️ Prediction Route Test
    def test_prediction_endpoint(self):
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
        self.assertIn(b"Result", response.data)  # HTML contains prediction

    # 3️ Metrics Endpoint Test
    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"app_request_total", response.data)
        self.assertIn(b"model_prediction_total", response.data)


if __name__ == "__main__":
    unittest.main()
