import logging
import pandas as pd
from src.utils import load_params
from src.utils import save_data
from sklearn.preprocessing import StandardScaler
from src.logger import configure_logger
from src.utils import save_model


logger = logging.getLogger(__name__)


def split_features_target(df: pd.DataFrame, features: list, target: str)-> pd.DataFrame:
    logger.info("Splitting dataset into features and target")
    
    X = df[features]
    y = df[target]

    return X, y


def compute_iqr_bounds(X: pd.DataFrame, cols: list) -> dict:
    """
    Compute IQR-based outlier bounds for selected columns.

    Returns:
        dict: {column: (lower_bound, upper_bound)}
    """
    bounds = {}

    for col in cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        bounds[col] = (lower, upper)

    return bounds

def apply_iqr_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Apply IQR bounds to cap outliers.
    """
    X_out = X.copy()

    for col, (lower, upper) in bounds.items():
        X_out[col] = X_out[col].clip(lower, upper)

    return X_out



def encode_binary_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary categorical columns in a compact way.
    """
    X = X.copy()

    X['gender']   = X['gender'].map({'male': 1, 'female': 0})
    X['diabetic'] = X['diabetic'].map({'Yes': 1, 'No': 0})
    X['smoker']   = X['smoker'].map({'Yes': 1, 'No': 0})

    return X


def main():
    try:

        logger.info("started data_preprocessin pipeline.....................\n")

        params = load_params(params_path = "params.yaml")
        features = params['data_preprocessing']['features']
        target = params['data_preprocessing']['target']
        outlier_cols = params['data_preprocessing']['outliers']
        num_cols = params['data_preprocessing']['num_cols']

        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv').dropna()
        test_data = pd.read_csv('./data/raw/test.csv').dropna()


        logging.info('data loaded.')

        train_X, train_y = split_features_target(train_data, features, target)
        test_X, test_y = split_features_target(test_data, features, target)

        iqr_bounds = compute_iqr_bounds(train_X, outlier_cols)
        train_X = apply_iqr_bounds(train_X, iqr_bounds)
        test_X = apply_iqr_bounds(test_X, iqr_bounds)

        train_X = encode_binary_columns(train_X)
        test_X  = encode_binary_columns(test_X)

        scaler = StandardScaler()

        train_X[num_cols] = scaler.fit_transform(train_X[num_cols])  # fit on train
        test_X[num_cols]  = scaler.transform(test_X[num_cols])       # apply to test


        save_model(scaler, "models/scaler.pkl")
        processed_train_path = save_data(train_X.join(train_y), "data/processed", "train.csv")
        processed_test_path  = save_data(test_X.join(test_y), "data/processed", "test.csv")

        logger.info("Processed data saved at %s and %s", processed_train_path, processed_test_path)

        logger.info("completed data_preprocessin pipeline.....................\n")


    except Exception as e:
        logger.error('Failed to complete the data_preprocessing pipeline: %s', e)
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main()