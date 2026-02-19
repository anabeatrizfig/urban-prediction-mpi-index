"""
Data Preprocessing Module for Urban Prediction MPI Index Pipeline.

This module handles data cleaning, missing value treatment, and outlier handling
for the H3 index dataset.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from .utils import load_config, logger, log_dataframe_info


class DataPreprocessor:
    """
    Class to preprocess H3 index data for MPI calculation.
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_path: str = None):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
            config (Dict[str, Any]): Configuration dictionary. If None, loads from config_path.
            config_path (str): Path to configuration file. If None, uses default.
        """
        if config is None:
            config = load_config(config_path)
        
        self.config = config
        self.preprocessing_config = config['preprocessing']
        
    def load_data(self, file_path: str) -> gpd.GeoDataFrame:
        """
        Load H3 index data from a GeoPackage file.
        
        Parameters:
            file_path (str): Path to the input GeoPackage file.
        
        Returns:
            gpd.GeoDataFrame: Loaded GeoDataFrame.
        """
        logger.info(f"Loading data from {file_path.split('/')[-1]}")
        gdf = gpd.read_file(file_path)
        log_dataframe_info(gdf, "Input data")
        return gdf
    
    def analyze_missing_values(self, gdf: gpd.GeoDataFrame, 
                                columns: List[str] = None) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            columns (List[str]): Columns to analyze. If None, uses config.
        
        Returns:
            pd.DataFrame: DataFrame with missing value statistics.
        """
        if columns is None:
            columns = self.preprocessing_config['missing_value_columns']
        
        # Filter to existing columns
        existing_cols = [col for col in columns if col in gdf.columns]
        
        missing = gdf[existing_cols].isnull().sum()
        missing_pct = (missing / len(gdf)) * 100
        
        result = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        })
        
        logger.info("Missing value analysis completed")
        return result.sort_values('missing_percentage', ascending=False)
    
    def analyze_missing_by_city(self, gdf: gpd.GeoDataFrame, 
                                 city_column: str = 'nm_city',
                                 columns: List[str] = None) -> pd.DataFrame:
        """
        Analyze missing values by city.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            city_column (str): Name of the city column.
            columns (List[str]): Columns to analyze. If None, uses config.
        
        Returns:
            pd.DataFrame: DataFrame with missing value percentages by city.
        """
        if columns is None:
            columns = self.preprocessing_config['missing_value_columns']
        
        # Filter to existing columns
        existing_cols = [col for col in columns if col in gdf.columns]
        
        missing_by_city = gdf.groupby(city_column)[existing_cols].agg(
            lambda x: x.isnull().mean() * 100
        )
        
        logger.info("Missing value analysis by city completed")
        return missing_by_city
    
    def filter_cities(self, gdf: gpd.GeoDataFrame, 
                      cities_to_exclude: List[str] = None,
                      city_column: str = 'nm_city') -> gpd.GeoDataFrame:
        """
        Filter out cities with excessive missing values.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            cities_to_exclude (List[str]): Cities to exclude. If None, uses config.
            city_column (str): Name of the city column.
        
        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame.
        """
        if cities_to_exclude is None:
            cities_to_exclude = self.preprocessing_config['cities_to_exclude']
        
        initial_count = len(gdf)
        gdf_filtered = gdf.query(f'{city_column} not in @cities_to_exclude').copy()
        
        logger.info(f"Filtered out {initial_count - len(gdf_filtered)} rows from "
                   f"{len(cities_to_exclude)} cities")
        
        return gdf_filtered
    
    def treat_outliers(self, gdf: gpd.GeoDataFrame, 
                       column: str,
                       method: str = 'quantile_cap',
                       quantile: float = 0.999,
                       fill_with: str = 'median') -> gpd.GeoDataFrame:
        """
        Treat outliers in a specific column.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            column (str): Column to treat.
            method (str): Outlier treatment method ('quantile_cap').
            quantile (float): Quantile threshold for capping.
            fill_with (str): Value to fill outliers with ('median').
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with treated outliers.
        """
        gdf = gdf.copy()
        
        if method == 'quantile_cap':
            threshold = gdf[column].quantile(quantile)
            fill_value = gdf[column].median() if fill_with == 'median' else fill_with
            
            outlier_count = (gdf[column] > threshold).sum()
            gdf[column] = gdf[column].apply(
                lambda x: x if x <= threshold else fill_value
            )
            
            logger.info(f"Treated {outlier_count} outliers in {column} "
                       f"(capped at {quantile} quantile = {threshold:.2f})")
        
        return gdf
    
    def fill_missing_values(self, gdf: gpd.GeoDataFrame,
                            columns: List[str] = None,
                            method: str = 'median') -> gpd.GeoDataFrame:
        """
        Fill missing values in specified columns.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            columns (List[str]): Columns to fill. If None, uses config.
            method (str): Fill method ('median', 'mean', 'mode').
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with filled values.
        """
        gdf = gdf.copy()
        
        if columns is None:
            columns = self.preprocessing_config['fill_median_columns']
        
        for col in columns:
            if col not in gdf.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
                
            missing_count = gdf[col].isnull().sum()
            
            if method == 'median':
                fill_value = gdf[col].median()
            elif method == 'mean':
                fill_value = gdf[col].mean()
            elif method == 'mode':
                fill_value = gdf[col].mode()[0]
            else:
                raise ValueError(f"Unknown fill method: {method}")
            
            gdf[col] = gdf[col].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing values in {col} with {method} ({fill_value:.2f})")
        
        return gdf
    
    def recalculate_income_columns(self, gdf: gpd.GeoDataFrame,
                                    minimum_wage: float = None) -> gpd.GeoDataFrame:
        """
        Recalculate income-related columns after filling missing values.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            minimum_wage (float): Minimum wage value. If None, uses config.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with recalculated income columns.
        """
        gdf = gdf.copy()
        
        if minimum_wage is None:
            minimum_wage = self.preprocessing_config['minimum_wage_value']
        
        # Recalculate minimum wage per household
        gdf['nr_minimum_wage_household'] = (
            gdf['nr_monthly_income_household'] / minimum_wage
        ).round(2)
        
        # Recalculate wage range category
        gdf['ds_minimum_wage_range_household'] = np.where(
            gdf['nr_minimum_wage_household'] <= 2, 
            '1. up to 2 minimum wages',
            np.where(
                gdf['nr_minimum_wage_household'] <= 5, 
                '2. from 2 to 5 minimum wages',
                np.where(
                    gdf['nr_minimum_wage_household'] > 5, 
                    '3. more than 5 minimum wages',
                    '4. no information'
                )
            )
        )
        
        logger.info("Recalculated income-related columns")
        return gdf
    
    def recreate_slope_column(self, gdf: gpd.GeoDataFrame,
                               slope_columns: List[str] = None) -> gpd.GeoDataFrame:
        """
        Recreate the numeric slope column based on slope class flags.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            slope_columns (List[str]): Slope class columns to use. If None, uses config.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with recreated slope column.
        """
        gdf = gdf.copy()
        
        if slope_columns is None:
            slope_columns = self.preprocessing_config['slope_class_columns']
        
        # Check if slope columns exist
        existing_slope_cols = [col for col in slope_columns if col in gdf.columns]
        
        if not existing_slope_cols:
            logger.warning("No slope class columns found, skipping slope recreation")
            return gdf
        
        # Recreate slope based on class flags
        np.random.seed(42)  # For reproducibility
        n = len(gdf)
        
        gdf['nr_slope'] = np.where(
            gdf.get('fl_slope_class_4', 0) == 1, 
            np.ceil(np.random.uniform(30, 45, size=n)),
            np.where(
                gdf.get('fl_slope_class_3', 0) == 1, 
                np.ceil(np.random.uniform(15, 30.1, size=n)),
                np.where(
                    gdf.get('fl_slope_class_2', 0) == 1, 
                    np.ceil(np.random.uniform(3, 15.1, size=n)),
                    np.where(
                        gdf.get('fl_slope_class_1', 0) == 1, 
                        np.ceil(np.random.uniform(0, 3.1, size=n)),
                        0
                    )
                )
            )
        )
        
        # Drop the slope class columns
        gdf = gdf.drop(columns=existing_slope_cols, errors='ignore')
        
        logger.info(f"Recreated nr_slope column and dropped {len(existing_slope_cols)} slope class columns")
        return gdf
    
    def preprocess(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Run the full preprocessing pipeline.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
        
        Returns:
            gpd.GeoDataFrame: Preprocessed GeoDataFrame.
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Filter cities with excessive missing values
        gdf = self.filter_cities(gdf)
        
        # Step 2: Treat outliers
        outlier_config = self.preprocessing_config.get('outliers', {})
        for col, settings in outlier_config.items():
            if col in gdf.columns:
                gdf = self.treat_outliers(
                    gdf, col,
                    method=settings.get('method', 'quantile_cap'),
                    quantile=settings.get('quantile', 0.999),
                    fill_with=settings.get('fill_with', 'median')
                )
        
        # Step 3: Fill missing values in numeric columns
        gdf = self.fill_missing_values(gdf)
        
        # Step 4: Recalculate income columns
        gdf = self.recalculate_income_columns(gdf)
        
        # Step 5: Recreate slope column
        gdf = self.recreate_slope_column(gdf)
        
        logger.info("Preprocessing pipeline completed")
        log_dataframe_info(gdf, "Preprocessed data")
        
        return gdf
    
    def save_data(self, gdf: gpd.GeoDataFrame, file_path: str) -> None:
        """
        Save preprocessed data to a GeoPackage file.
        
        Parameters:
            gdf (gpd.GeoDataFrame): GeoDataFrame to save.
            file_path (str): Path to output file.
        """
        gdf.to_file(file_path, driver='GPKG')
        logger.info(f"Data saved to {file_path.split('/')[-1]}")


def preprocess_data(input_path: str, 
                    output_path: str = None,
                    config_path: str = None) -> gpd.GeoDataFrame:
    """
    Main function to preprocess data.
    
    Parameters:
        input_path (str): Path to input GeoPackage file.
        output_path (str): Path to output file. If None, doesn't save.
        config_path (str): Path to configuration file.
    
    Returns:
        gpd.GeoDataFrame: Preprocessed GeoDataFrame.
    """
    preprocessor = DataPreprocessor(config_path=config_path)
    
    gdf = preprocessor.load_data(input_path)
    gdf_preprocessed = preprocessor.preprocess(gdf)
    
    if output_path:
        preprocessor.save_data(gdf_preprocessed, output_path)
    
    return gdf_preprocessed


def main():
    """Entry point for the preprocessing CLI."""
    import argparse
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Preprocess H3 index data")
    parser.add_argument("--input", "-i", type=str, help="Input file path")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    
    args = parser.parse_args()
    
    # Load config to get default file paths
    config = load_config(args.config)
    data_config = config['data']
    
    # Use environment variable for input (original data location)
    local_path = os.getenv('LOCAL_FILE_PATH', '')
    
    # Project root and data folder for outputs
    project_root = Path(__file__).parent.parent
    data_folder = project_root / data_config['output_folder']
    data_folder.mkdir(exist_ok=True)
    
    input_path = args.input or f"{local_path}{data_config['input_file']}"
    output_path = args.output or str(data_folder / data_config['preprocessed_file'])
    
    gdf = preprocess_data(input_path, output_path, args.config)
    print(f"Preprocessing complete. Output shape: {gdf.shape}")


if __name__ == "__main__":
    main()
