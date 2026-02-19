"""
End-to-End Pipeline for Urban Prediction MPI Index.

This module provides a unified interface to run the complete pipeline
from raw data to MPI-enriched output.
"""

import geopandas as gpd
from typing import Dict, Any, Optional
from pathlib import Path

from .preprocessing import DataPreprocessor, preprocess_data
from .feature_engineering import MPICalculator, compute_mpi_features
from .utils import load_config, logger


class MIPPipeline:
    """
    End-to-end pipeline for MPI index computation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline.
        
        Parameters:
            config_path (str): Path to configuration file.
        """
        self.config = load_config(config_path)
        self.preprocessor = DataPreprocessor(config=self.config)
        self.mpi_calculator = MPICalculator(config=self.config)
        
    def run(self, 
            input_path: str,
            output_path: str = None,
            save_intermediate: bool = False,
            intermediate_path: str = None) -> gpd.GeoDataFrame:
        """
        Run the complete pipeline.
        
        Parameters:
            input_path (str): Path to raw input data.
            output_path (str): Path to save final output.
            save_intermediate (bool): Whether to save preprocessed data.
            intermediate_path (str): Path for intermediate file.
        
        Returns:
            gpd.GeoDataFrame: Final GeoDataFrame with MPI features.
        """
        logger.info("=" * 60)
        logger.info("Starting MPI Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load raw data
        logger.info("\n[Step 1/3] Loading raw data...")
        gdf = self.preprocessor.load_data(input_path)
        
        # Step 2: Preprocess
        logger.info("\n[Step 2/3] Preprocessing data...")
        gdf_preprocessed = self.preprocessor.preprocess(gdf)
        
        if save_intermediate and intermediate_path:
            self.preprocessor.save_data(gdf_preprocessed, intermediate_path)
        
        # Step 3: Compute MPI
        logger.info("\n[Step 3/3] Computing MPI features...")
        gdf_final = self.mpi_calculator.compute_all_mpi(gdf_preprocessed)
        
        # Save final output
        if output_path:
            gdf_final.to_file(output_path, driver='GPKG')
            logger.info(f"Final output saved to {output_path}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)
        
        summary = self.mpi_calculator.get_mpi_summary(gdf_final)
        logger.info(f"\nMPI Summary:\n{summary}")
        
        return gdf_final
    
    def run_preprocessing_only(self, 
                               input_path: str, 
                               output_path: str = None) -> gpd.GeoDataFrame:
        """
        Run only the preprocessing step.
        
        Parameters:
            input_path (str): Path to raw input data.
            output_path (str): Path to save preprocessed output.
        
        Returns:
            gpd.GeoDataFrame: Preprocessed GeoDataFrame.
        """
        return preprocess_data(input_path, output_path)
    
    def run_feature_engineering_only(self,
                                      input_path: str,
                                      output_path: str = None,
                                      compute_per_city: bool = True) -> gpd.GeoDataFrame:
        """
        Run only the feature engineering step.
        
        Parameters:
            input_path (str): Path to preprocessed input data.
            output_path (str): Path to save output with MPI.
            compute_per_city (bool): Whether to compute per-city MPI.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with MPI features.
        """
        return compute_mpi_features(input_path, output_path, 
                                    compute_per_city=compute_per_city)


def run_pipeline(input_path: str,
                 output_path: str = None,
                 config_path: str = None,
                 save_intermediate: bool = False,
                 intermediate_path: str = None) -> gpd.GeoDataFrame:
    """
    Convenience function to run the complete pipeline.
    
    Parameters:
        input_path (str): Path to raw input data.
        output_path (str): Path to save final output.
        config_path (str): Path to configuration file.
        save_intermediate (bool): Whether to save preprocessed data.
        intermediate_path (str): Path for intermediate file.
    
    Returns:
        gpd.GeoDataFrame: Final GeoDataFrame with MPI features.
    """
    pipeline = MIPPipeline(config_path=config_path)
    return pipeline.run(
        input_path, 
        output_path, 
        save_intermediate, 
        intermediate_path
    )


def main():
    """Entry point for the pipeline CLI."""
    import argparse
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Run the Urban Prediction MPI Index Pipeline"
    )
    parser.add_argument("--input", "-i", type=str, help="Input file path")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    parser.add_argument("--intermediate", type=str, 
                        help="Path to save intermediate preprocessed file")
    parser.add_argument("--step", type=str, 
                        choices=['all', 'preprocess', 'features'],
                        default='all',
                        help="Which step(s) to run")
    
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
    output_path = args.output or str(data_folder / data_config['output_file'])
    intermediate_path = args.intermediate or str(data_folder / data_config['preprocessed_file'])
    
    pipeline = MIPPipeline(config_path=args.config)
    
    if args.step == 'all':
        gdf = pipeline.run(
            input_path, 
            output_path,
            save_intermediate=True,
            intermediate_path=intermediate_path
        )
    elif args.step == 'preprocess':
        gdf = pipeline.run_preprocessing_only(input_path, intermediate_path)
    elif args.step == 'features':
        gdf = pipeline.run_feature_engineering_only(intermediate_path, output_path)
    
    print(f"\nPipeline '{args.step}' complete. Final shape: {gdf.shape}")


if __name__ == "__main__":
    main()
