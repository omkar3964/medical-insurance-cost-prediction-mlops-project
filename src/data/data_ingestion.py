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
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    duplicated_rows = df.duplicated().sum()
    columns_names = df.columns
    num_rows, num_cols = df.shape
    missing_values = df.isnull().sum().to_dict()

    logger.info(
        "All required columns are present.\n"
        "Data shape: %d rows, %d columns\n"
        "Columns: %s\n"
        "Duplicated rows: %d\n"
        "Missing values per column: %s",
        num_rows, num_cols, list(columns_names), duplicated_rows, missing_values
    )


def main():
    try:
        logger.info("Started data_ingestion pipeline...")

        # Load parameters
        params = load_params(params_path="params.yaml")
        test_size = params['data_ingestion']['test_size']
        required_columns = params['data_ingestion']['required_columns']

        # Load dataset
        df = load_data(Path("notebooks") / "insurance.csv")
        validate_columns(df, required_columns)

        # Split dataset
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        # Ensure directories exist
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        train_path = save_data(train_data, raw_dir, "train.csv")
        test_path = save_data(test_data, raw_dir, "test.csv")
        logger.info("Raw data saved at %s and %s", train_path, test_path)

        logger.info("Completed data_ingestion pipeline.")

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


