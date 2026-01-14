import logging
import mlflow
import dagshub
from src.logger import configure_logger
from src.utils import load_data, load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import json
logger = logging.getLogger(__name__)


# local 
mlflow.set_tracking_uri('https://dagshub.com/omalbhare/medical-insurance-cost-prediction-mlops-project.mlflow')
dagshub.init(repo_owner='omalbhare', repo_name='medical-insurance-cost-prediction-mlops-project', mlflow=True)

# production 


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return the evaluation metrics."""
    try:
        logger.info('started Model evaluation...........')
        y_pred = model.predict(X_test)
                
        metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }
        logger.info('Model evaluation metrics calculated')
        return metrics
    
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise 
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("regression model evaluation")
    with mlflow.start_run() as run:
        try:
            model = load_model('models/model.pkl')
            model = load_model('models/model.pkl')

            test_data = load_data('data/processed/test.csv')

            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            metrics = evaluate_model(model, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')

             # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.log_param("test_rows", X_test.shape[0])
            mlflow.log_param("test_features", X_test.shape[1])
            mlflow.log_dict({"features": test_data.columns[:-1].tolist()}, "features.json")

            
            # Log model parameters to MLflow
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact("models/scaler.pkl", artifact_path="preprocessor")

            save_model_info(run.info.run_id, f"runs:/{run.info.run_id}/model",  "reports/experiment_info.json")


            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            raise


if __name__ == '__main__':
    main()

