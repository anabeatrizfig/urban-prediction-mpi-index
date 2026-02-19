"""
Feature Engineering Module for Urban Prediction MPI Index Pipeline.

This module implements the Mazziotta-Pareto Index (MPI) calculation
for urban comfort analysis based on multiple indicators.

Reference:
    Developing the urban comfort index: Advancing liveability analytics 
    with a multidimensional approach and explainable artificial intelligence
    (Binyu Lei, Pengyuan Liu, Xiucheng Liang, Yingwei Yan, Filip Biljecki)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from .utils import load_config, logger, log_dataframe_info, get_mpi_features_and_polarity


class MPICalculator:
    """
    Class to calculate the Mazziotta-Pareto Index (MPI) for urban analysis.
    
    The MPI method aggregates multiple indicators at spatial units and 
    summarizes their impact into a single index. It penalizes imbalances 
    among positive and negative indicators.
    
    Formula:
        MPI_i = MZ_i - SZ_i * cvZ_i
        
    Where:
        - MZ_i is the mean of standardized values for unit i
        - SZ_i is the standard deviation of standardized values for unit i
        - cvZ_i is the coefficient of variation (SZ_i / MZ_i)
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_path: str = None):
        """
        Initialize the MPI Calculator.
        
        Parameters:
            config (Dict[str, Any]): Configuration dictionary. If None, loads from config_path.
            config_path (str): Path to configuration file. If None, uses default.
        """
        if config is None:
            config = load_config(config_path)
        
        self.config = config
        self.mpi_config = config['mpi']
        self.output_config = config['output']
        
        # Extract features and polarities from config
        self.features, self.polarities = get_mpi_features_and_polarity(config)
        
    def standardize_indicators(self, data: pd.DataFrame, 
                                polarity: List[str]) -> pd.DataFrame:
        """
        Standardize indicators using the MPI standardization formula.
        
        Formula:
            Z_ij = 100 + (x_ij - M_xj) / S_xj * 10  (for positive polarity)
            Z_ij = 100 - (x_ij - M_xj) / S_xj * 10  (for negative polarity)
        
        Parameters:
            data (pd.DataFrame): DataFrame with indicator columns.
            polarity (List[str]): List of '+' or '-' for each indicator.
        
        Returns:
            pd.DataFrame: Standardized DataFrame.
        """
        standardized = pd.DataFrame(index=data.index)
        
        for col, pol in zip(data.columns, polarity):
            mean = data[col].mean()
            std = data[col].std()
            
            if std == 0:
                logger.warning(f"Column {col} has zero standard deviation, using 100 as standardized value")
                standardized[col] = 100
                continue
            
            if pol == '+':
                standardized[col] = 100 + ((data[col] - mean) / std) * 10
            elif pol == '-':
                standardized[col] = 100 - ((data[col] - mean) / std) * 10
            else:
                raise ValueError(f"Polarity must be '+' or '-', got '{pol}'")
        
        return standardized
    
    def compute_mpi(self, data: pd.DataFrame, 
                    polarity: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute the Mazziotta-Pareto Index (MPI).
        
        Parameters:
            data (pd.DataFrame): DataFrame with indicator columns.
            polarity (List[str]): List of '+' or '-' for each indicator.
        
        Returns:
            Tuple[pd.Series, pd.DataFrame]: MPI values and standardized data.
        """
        # Step 1: Standardize the indicators
        standardized_data = self.standardize_indicators(data, polarity)
        
        # Step 2: Compute MZ_i, SZ_i, and cvZ_i for each unit
        MZ_i = standardized_data.mean(axis=1)
        SZ_i = standardized_data.std(axis=1)
        
        # Avoid division by zero
        cvZ_i = np.where(MZ_i != 0, SZ_i / MZ_i, 0)
        
        # Step 3: Compute MPI
        MPI = MZ_i - SZ_i * cvZ_i
        
        logger.info(f"MPI computed for {len(MPI)} units")
        logger.info(f"MPI statistics - Mean: {MPI.mean():.2f}, Std: {MPI.std():.2f}, "
                   f"Min: {MPI.min():.2f}, Max: {MPI.max():.2f}")
        
        return MPI, standardized_data
    
    def classify_mpi(self, mpi_values: pd.Series) -> pd.Series:
        """
        Classify MPI values based on threshold.
        
        Parameters:
            mpi_values (pd.Series): MPI values.
        
        Returns:
            pd.Series: Classification labels.
        """
        threshold = self.mpi_config['threshold']
        labels = self.mpi_config['labels']
        
        classification = np.where(
            mpi_values >= threshold,
            labels['above_threshold'],
            np.where(
                mpi_values < threshold,
                labels['below_threshold'],
                labels['no_data']
            )
        )
        
        return pd.Series(classification, index=mpi_values.index)
    
    def compute_mpi_global(self, gdf: gpd.GeoDataFrame,
                           features: List[str] = None,
                           polarity: List[str] = None) -> gpd.GeoDataFrame:
        """
        Compute MPI globally (across all data).
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            features (List[str]): Feature columns. If None, uses config.
            polarity (List[str]): Polarities. If None, uses config.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with MPI columns added.
        """
        gdf = gdf.copy()
        
        if features is None:
            features = self.features
        if polarity is None:
            polarity = self.polarities
        
        # Filter to existing features
        existing_features = [f for f in features if f in gdf.columns]
        existing_polarity = [p for f, p in zip(features, polarity) if f in gdf.columns]
        
        if len(existing_features) != len(features):
            missing = set(features) - set(existing_features)
            logger.warning(f"Missing features: {missing}")
        
        # Extract feature data
        data = gdf[existing_features]
        
        # Compute MPI
        mpi_values, standardized = self.compute_mpi(data, existing_polarity)
        
        # Add results to GeoDataFrame
        mpi_col = self.output_config['mpi_index_column']
        flag_col = self.output_config['mpi_flag_column']
        
        gdf[mpi_col] = mpi_values
        gdf[flag_col] = self.classify_mpi(mpi_values)
        
        logger.info(f"Global MPI computed - Above threshold: "
                   f"{(mpi_values >= self.mpi_config['threshold']).sum()}, "
                   f"Below threshold: {(mpi_values < self.mpi_config['threshold']).sum()}")
        
        return gdf
    
    def compute_mpi_per_city(self, gdf: gpd.GeoDataFrame,
                             city_column: str = 'nm_city',
                             h3_column: str = 'h3_index',
                             features: List[str] = None,
                             polarity: List[str] = None) -> gpd.GeoDataFrame:
        """
        Compute MPI separately for each city.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            city_column (str): Name of the city column.
            h3_column (str): Name of the H3 index column.
            features (List[str]): Feature columns. If None, uses config.
            polarity (List[str]): Polarities. If None, uses config.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with per-city MPI columns added.
        """
        gdf = gdf.copy()
        
        if features is None:
            features = self.features
        if polarity is None:
            polarity = self.polarities
        
        # Filter to existing features
        existing_features = [f for f in features if f in gdf.columns]
        existing_polarity = [p for f, p in zip(features, polarity) if f in gdf.columns]
        
        cities = gdf[city_column].unique().tolist()
        logger.info(f"Computing MPI for {len(cities)} cities")
        
        results = []
        
        for city in cities:
            logger.info(f"Processing city: {city}")
            city_mask = gdf[city_column] == city
            city_data = gdf.loc[city_mask, existing_features]
            
            # Compute MPI for this city
            mpi_values, _ = self.compute_mpi(city_data, existing_polarity)
            
            city_df = pd.DataFrame({
                h3_column: gdf.loc[city_mask, h3_column],
                city_column: city,
                'mpi_per_city': mpi_values
            })
            
            results.append(city_df)
        
        # Concatenate results
        mpi_df = pd.concat(results, ignore_index=True)
        
        # Merge with original GeoDataFrame
        mpi_col_city = self.output_config['mpi_index_per_city_column']
        flag_col_city = self.output_config['mpi_flag_per_city_column']
        
        gdf = gdf.merge(
            mpi_df[[h3_column, city_column, 'mpi_per_city']],
            on=[h3_column, city_column],
            how='left'
        )
        
        gdf = gdf.rename(columns={'mpi_per_city': mpi_col_city})
        gdf[flag_col_city] = self.classify_mpi(gdf[mpi_col_city])
        
        logger.info(f"Per-city MPI computed for all cities")
        
        return gdf
    
    def compute_all_mpi(self, gdf: gpd.GeoDataFrame,
                        city_column: str = 'nm_city',
                        h3_column: str = 'h3_index') -> gpd.GeoDataFrame:
        """
        Compute both global and per-city MPI.
        
        Parameters:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            city_column (str): Name of the city column.
            h3_column (str): Name of the H3 index column.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with all MPI columns added.
        """
        logger.info("Computing global MPI")
        gdf = self.compute_mpi_global(gdf)
        
        logger.info("Computing per-city MPI")
        gdf = self.compute_mpi_per_city(gdf, city_column, h3_column)
        
        return gdf
    
    def get_mpi_summary(self, gdf: gpd.GeoDataFrame, 
                        city_column: str = 'nm_city') -> pd.DataFrame:
        """
        Generate a summary of MPI results by city.
        
        Parameters:
            gdf (gpd.GeoDataFrame): GeoDataFrame with MPI columns.
            city_column (str): Name of the city column.
        
        Returns:
            pd.DataFrame: Summary statistics by city.
        """
        mpi_col = self.output_config['mpi_index_column']
        mpi_col_city = self.output_config['mpi_index_per_city_column']
        threshold = self.mpi_config['threshold']
        
        summary = gdf.groupby(city_column).agg({
            mpi_col: ['mean', 'std', 'min', 'max'],
            mpi_col_city: ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Add count above/below threshold
        counts = gdf.groupby(city_column).apply(
            lambda x: pd.Series({
                'count_total': len(x),
                'count_above_threshold_global': (x[mpi_col] >= threshold).sum(),
                'count_above_threshold_city': (x[mpi_col_city] >= threshold).sum()
            })
        )
        
        summary = summary.join(counts)
        
        return summary


def compute_mpi_features(input_path: str,
                         output_path: str = None,
                         config_path: str = None,
                         compute_per_city: bool = True) -> gpd.GeoDataFrame:
    """
    Main function to compute MPI features.
    
    Parameters:
        input_path (str): Path to input GeoPackage file.
        output_path (str): Path to output file. If None, doesn't save.
        config_path (str): Path to configuration file.
        compute_per_city (bool): Whether to compute per-city MPI.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with MPI features.
    """
    calculator = MPICalculator(config_path=config_path)
    
    logger.info(f"Loading data from {input_path}")
    gdf = gpd.read_file(input_path)
    log_dataframe_info(gdf, "Input data")
    
    if compute_per_city:
        gdf = calculator.compute_all_mpi(gdf)
    else:
        gdf = calculator.compute_mpi_global(gdf)
    
    if output_path:
        gdf.to_file(output_path, driver='GPKG')
        logger.info(f"Data saved to {output_path}")
    
    # Print summary
    summary = calculator.get_mpi_summary(gdf)
    logger.info(f"\nMPI Summary by City:\n{summary}")
    
    return gdf


def main():
    """Entry point for the feature engineering CLI."""
    import argparse
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Compute MPI features for H3 index data")
    parser.add_argument("--input", "-i", type=str, help="Input file path")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    parser.add_argument("--no-per-city", action="store_true", 
                        help="Skip per-city MPI calculation")
    
    args = parser.parse_args()
    
    # Load config to get default file paths
    config = load_config(args.config)
    data_config = config['data']
    
    # Project root and data folder
    project_root = Path(__file__).parent.parent
    data_folder = project_root / data_config['output_folder']
    data_folder.mkdir(exist_ok=True)
    
    # Input from data folder (preprocessed), output to data folder
    input_path = args.input or str(data_folder / data_config['preprocessed_file'])
    output_path = args.output or str(data_folder / data_config['output_file'])
    
    gdf = compute_mpi_features(
        input_path, 
        output_path, 
        args.config,
        compute_per_city=not args.no_per_city
    )
    
    print(f"Feature engineering complete. Output shape: {gdf.shape}")


if __name__ == "__main__":
    main()
