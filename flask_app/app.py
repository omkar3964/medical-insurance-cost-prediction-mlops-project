import joblib
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import dagshub
from load_model import load_model_and_scaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Flask App
app = Flask(__name__)


# Prometheus Metrics
registry = CollectorRegistry()

# HTTP request metrics
REQUEST_COUNT = Counter("app_request_total", "Total number of HTTP requests",  ["method", "endpoint"],  registry=registry,)
REQUEST_LATENCY = Histogram( "app_request_latency_seconds",  "HTTP request latency in seconds",  ["endpoint"],  registry=registry,)

# Regression model metrics
PREDICTION_COUNT = Counter( "model_prediction_total", "Total number of regression predictions", registry=registry)
PREDICTION_VALUE = Histogram("model_prediction_value", "Distribution of regression prediction values", buckets=(0, 1000, 5000, 10000, 20000, 50000), registry=registry,)
PREDICTION_LATENCY = Histogram("model_prediction_latency_seconds", "Latency of regression model predictions", registry=registry)

# Load at startup
MODEL_NAME = "InsuranceCostModel"
model, scaler = load_model_and_scaler(MODEL_NAME, stage="Production")


# -------------------------------
# Feature Preparation
# -------------------------------
def prepare_input_features(form_data, scaler):
    age = int(form_data["age"])
    bmi = float(form_data["bmi"])
    bloodpressure = int(form_data["bloodpressure"])

    gender = 1 if form_data["gender"] == "male" else 0
    diabetic = 1 if form_data["diabetic"] == "yes" else 0
    children = int(form_data["children"])
    smoker = 1 if form_data["smoker"] == "yes" else 0

    # Create DataFrame with SAME column names used during training
    numeric_df = pd.DataFrame( [[age, bmi, bloodpressure]],  columns=["age", "bmi", "bloodpressure"] )

    # Scale using DataFrame (no warning)
    scaled_numeric = scaler.transform(numeric_df)

    # Final feature DataFrame (match training order)
    features_df = pd.DataFrame([[scaled_numeric[0, 0], gender, scaled_numeric[0, 1], scaled_numeric[0, 2],diabetic,children,smoker ]],
      columns=["age","gender","bmi","bloodpressure","diabetic","children","smoker"]
    )

    return features_df



# Routes
@app.route("/")
def home():
    start_time = time.time()
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    try:
        # Prepare features
        features = prepare_input_features(request.form, scaler)

        # Predict
        prediction_start = time.time()
        y_pred = model.predict(features)
        PREDICTION_LATENCY.observe(time.time() - prediction_start)
        PREDICTION_COUNT.inc()
        PREDICTION_VALUE.observe(float(y_pred[0]))

        result = round(float(y_pred[0]), 2)
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        print(result)
        return render_template("index.html", result=result)

    except Exception as e:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return render_template("index.html", result=f"Error: {e}")


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


# Run App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
