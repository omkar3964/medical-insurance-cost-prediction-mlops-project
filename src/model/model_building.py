import logging
from src.logger import configure_logger
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from src.utils import load_params, load_data
from src.utils import save_model
logger = logging.getLogger(__name__)



from sklearn.ensemble import GradientBoostingRegressor

def train_model(model_class, training_params, train_data):
    """    Generic training function for regression models    """
    logger.info("starting training of GradientBoostingRegressor model\n")

    target_col = "claim"

    X_train = train_data.drop(columns = [target_col])
    y_train = train_data[target_col]

    # Initialize model with provided parameters
    model = model_class(
        learning_rate=training_params.get("learning_rate", 0.1),
        max_depth=training_params.get("max_depth", 3),
        n_estimators=training_params.get("n_estimators", 100),
        subsample=training_params.get("subsample", 1.0),
        random_state=training_params.get("random_state", 42)
    )

    # Train model
    model.fit(X_train, y_train)
    logger.info("completed training of GradientBoostingRegressor model\n")

    return model


def main():
    try:
        logger.info("started model_building pipeline.....................\n")

        logger.info("loading model_building parameter\n")
        params = load_params(params_path='params.yaml')
        training_params = params['model_building']['parameters']
        train_data = load_data('./data/processed/train.csv')

        model = train_model(GradientBoostingRegressor, training_params, train_data)

        save_model(model, 'models/model.pkl')

        logger.info("completed model_building pipeline.....................\n")
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")




if __name__ == "__main__":
    main()