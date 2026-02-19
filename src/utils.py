"""
Utility functions for the Urban Prediction MPI Index Pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters:
        config_path (str): Path to the configuration file.
                          If None, uses default path relative to this file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path.name}")
    return config


def get_mpi_features_and_polarity(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extract MPI features and their polarities from configuration.
    
    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    
    Returns:
        Tuple[List[str], List[str]]: Lists of feature names and their polarities.
    """
    features = []
    polarities = []
    
    for feature_name, feature_config in config['mpi']['features'].items():
        features.append(feature_name)
        polarities.append(feature_config['polarity'])
    
    return features, polarities


def validate_dataframe_columns(df, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    
    Parameters:
        df: DataFrame to validate.
        required_columns (List[str]): List of required column names.
    
    Returns:
        bool: True if all columns are present, raises ValueError otherwise.
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("All required columns are present in the DataFrame")
    return True


def log_dataframe_info(df, name: str = "DataFrame") -> None:
    """
    Log basic information about a DataFrame.
    
    Parameters:
        df: DataFrame to log information about.
        name (str): Name of the DataFrame for logging.
    """
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {list(df.columns)}")
    
    # Log missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        logger.info(f"{name} columns with missing values: {dict(missing_cols)}")
    else:
        logger.info(f"{name} has no missing values")
