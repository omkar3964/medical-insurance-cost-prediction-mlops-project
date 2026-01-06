import json
import mlflow
import logging
from mlflow.tracking import MlflowClient
import dagshub
from src.logger import configure_logger
from src.utils import load_params
from mlflow.exceptions import MlflowException


logger = logging.getLogger(__name__)


mlflow.set_tracking_uri('https://dagshub.com/omalbhare/medical-insurance-cost-prediction-mlops-project.mlflow')
dagshub.init(repo_owner='omalbhare', repo_name='medical-insurance-cost-prediction-mlops-project', mlflow=True)

client = MlflowClient()




# -------------------------------
# Utility Functions
# -------------------------------
def load_json(path: str) -> dict:
    """ loading model information """

    try:
        logger.info('loading model information.')
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error('Error occurred while loading model information.: %s', e)
        raise


def model_exists(model_name: str) -> bool:
    logger.info("Checking if registered model exists")
    try:
        client.get_registered_model(model_name)
        logger.info("Registered model exists")
        return True
    except MlflowException:
        logger.info("Registered model does not exist yet")
        return False


def meets_acceptance(metrics: dict, ACCEPTANCE_CRITERIA:dict) -> bool:
    return ( metrics["r2"] >= ACCEPTANCE_CRITERIA["min_r2"] and metrics["rmse"] <= ACCEPTANCE_CRITERIA["max_rmse"] )


def get_production_metrics(model_name: str):
    """ getting latest version of model for updation from model registry """
    try:
        logger.info("getting latest version of model for updation from model registry.")
        versions = client.get_latest_versions(model_name, stages=["Production"] )

        if not versions:
            return None

        run_id = versions[0].run_id
        return client.get_run(run_id).data.metrics
    except Exception as e:
        logger.error('Error occurred while getting latest version of model from model registry.: %s', e)
        raise


def is_better_than_production(new: dict, prod: dict) -> bool:
    return (  new["r2"] > prod["r2"] and new["rmse"] < prod["rmse"]   )


def decide_stage(metrics: dict, prod_metrics: dict | None, ACCEPTANCE_CRITERIA : dict):
    """
    Decision policy:
    - Reject if below acceptance
    - If no production model → Staging
    - If better than production → Production
    - Else → Staging
    """
    try:
        logger.info('determining status of model')
        if not meets_acceptance(metrics, ACCEPTANCE_CRITERIA):
            return None

        if prod_metrics is None:
            logger.info("No production model exists. Promoting first accepted model to Production.")
            return "Production"


        if is_better_than_production(metrics, prod_metrics):
            return "Production"

        return "Staging"
        
    except Exception as e:
        logger.error('Error occurred while deciding stage of model.: %s', e)
        raise


def main():

    try:
        logger.info("started model registry pipeline.....................\n")

        metrics = load_json("reports/metrics.json")
        run_info = load_json("reports/experiment_info.json")

        params = load_params('params.yaml')
        MODEL_NAME = params["register_model"]["MODEL_NAME"]
        ACCEPTANCE_CRITERIA = params["register_model"]["ACCEPTANCE_CRITERIA"]

        prod_metrics = None
        if model_exists(MODEL_NAME):
            prod_metrics = get_production_metrics(MODEL_NAME)

        target_stage = decide_stage(metrics, prod_metrics, ACCEPTANCE_CRITERIA)

        if target_stage is None:
            logger.warning("Model rejected — acceptance criteria failed")
            client.set_tag(run_info["run_id"], "registry_status", "rejected")
            return


        # Register model
        mv = mlflow.register_model( model_uri=run_info["model_path"], name=MODEL_NAME  )

        # Promote to correct stage
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage=target_stage,
            archive_existing_versions=(target_stage == "Production")
        )

        # Tags for audit & tracking
        client.set_model_version_tag( MODEL_NAME, mv.version, "status", "accepted"    )
        client.set_model_version_tag(  MODEL_NAME, mv.version, "stage_reason", target_stage.lower() )

        print(f"Model registered and moved to {target_stage}")

        logger.info("completed model registry pipeline.....................\n")
    except Exception as e:
        logger.error('Failed to complete the model registry process: %s', e)
        print(f"Error: {e}")

    
if __name__ == "__main__":
    main()
