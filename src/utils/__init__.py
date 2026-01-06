import yaml
from pathlib import Path
import pandas as pd
import logging
import joblib
logger = logging.getLogger(__name__)


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML file

    Returns:
        dict: Parsed YAML parameters
    """
    logger.info("reading data ingestion params from params.yaml file")
    try:
        path = Path(params_path)

        if not path.exists():
            raise FileNotFoundError(f"Parameters file not found: {path}")

        with path.open("r") as file:
            params = yaml.safe_load(file)

        if params is None:
            raise ValueError(f"Empty YAML file: {path}")

        logger.debug("Parameters loaded successfully from %s", path)
        return params

    except yaml.YAMLError as e:
        logger.error("YAML parsing error in %s: %s", params_path, e)
        raise

    except Exception:
        logger.exception("Failed to load parameters from %s", params_path)
        raise


def load_data(data_path: str, file_type: str = "csv") -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.

    Args:
        data_path (str): Path to the data file
        file_type (str): Type of the file: 'csv', 'json', or 'excel'. Default is 'csv'.

    Returns:
        pd.DataFrame: Loaded data
    """
    logger.info("Loading data from file: %s", data_path)
    try:
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if file_type.lower() == "csv":
            df = pd.read_csv(path)
        elif file_type.lower() == "json":
            df = pd.read_json(path)
        elif file_type.lower() in ["xls", "xlsx", "excel"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if df.empty:
            raise ValueError(f"Loaded data is empty: {path}")

        logger.debug("Data loaded successfully from %s", path)
        return df

    except pd.errors.ParserError as e:
        logger.error("Parsing error while reading %s: %s", data_path, e)
        raise

    except Exception:
        logger.exception("Failed to load data from %s", data_path)
        raise


def save_data(df: pd.DataFrame, folder_path: str, file_name: str) -> Path:
    """
    Save a DataFrame to a CSV file in the specified folder. Creates the folder if it doesn't exist.

    Args:
        df (pd.DataFrame): DataFrame to save
        folder_path (str): Folder path where CSV will be saved
        file_name (str): Name of the CSV file (e.g., 'train.csv')

    Returns:
        Path: Full path to the saved CSV file
    """
    try:
        # Ensure folder exists
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        # Full path to the file
        file_path = folder / file_name

        # Save DataFrame
        df.to_csv(file_path, index=False)
        logger.info("Data saved successfully at %s", file_path)

        return file_path

    except Exception as e:
        logger.exception("Failed to save data to %s/%s: %s", folder_path, file_name, e)
        raise



def save_model(model, file_path: str) -> None:
    """Save the trained model to disk using joblib."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, file_path)

        logger.info("Model successfully saved at %s", file_path)

    except Exception as e:
        logger.exception("Failed to save model at %s", file_path)
        raise