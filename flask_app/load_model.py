import dagshub
import mlflow
import joblib
def load_model_and_scaler(model_name: str, stage: str = "Production"):
    """
    Load MLflow pyfunc model and scaler artifact
    """
    try:
        mlflow.set_tracking_uri(
            "https://dagshub.com/omalbhare/medical-insurance-cost-prediction-mlops-project.mlflow"
        )
        dagshub.init(
            repo_owner="omalbhare",
            repo_name="medical-insurance-cost-prediction-mlops-project",
            mlflow=True,
        )

        # Load model
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Download scaler from run
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(name=model_name, stages=[stage])[0]
        run_id = latest_version_info.run_id
        scaler_path = mlflow.artifacts.download_artifacts(artifact_path="preprocessor/scaler.pkl", run_id=run_id)

        with open(scaler_path, "rb") as f:
            scaler = joblib.load(f)

        return model, scaler

    except Exception as e:
        raise RuntimeError(f"Failed to load model and scaler: {e}")


