import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from src.logger import configure_logger
from src.utils import load_params, load_data, save_data

# Configure logger
configure_logger()
logger = logging.getLogger(__name__)


def validate_columns(df: pd.DataFrame, required_columns: list) -> None:
    """
    Validate if all required columns are present in the DataFrame and log basic information about the data.

    Args:
        df (pd.DataFrame): Loaded dataset
        required_columns (list): List of required column names

    Raises:
        ValueError: If any required column is missing
    """
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Log basic information about data
    duplicated_rows = df.duplicated().sum()
    columns_names = df.columns
    num_rows, num_cols = df.shape
    missing_values = df.isnull().sum().to_dict()

    logger.info("All required columns are present.\n"
        "Data shape: %d rows, %d columns\n"
        "Columns: %s\n"
        "Duplicated rows: %d\n"
        "Missing values per column: %s",
        num_rows, num_cols, list(columns_names), duplicated_rows, missing_values
    )



def main():
    logger.info("started data_ingestion pipeline.....................\n")

    params = load_params(params_path = "params.yaml")
    test_size = params['data_ingestion']['test_size']
    required_columns = params['data_ingestion']['required_columns']
    
    df = load_data('notebooks\insurance.csv')
    validate_columns(df, required_columns)

    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    train_path = save_data(train_data, "data/raw", "train.csv")
    test_path = save_data(test_data, "data/raw", "test.csv")


    logger.info("completed data_ingestion  pipeline.....................\n")

if __name__ == "__main__":
    main()