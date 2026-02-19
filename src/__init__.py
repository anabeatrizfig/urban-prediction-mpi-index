"""
Urban Prediction MPI Index Pipeline

A pipeline for preprocessing geospatial data and computing the 
Mazziotta-Pareto Index (MPI) for urban analysis.
"""

from .preprocessing import DataPreprocessor, preprocess_data
from .feature_engineering import MPICalculator, compute_mpi_features
from .utils import load_config, get_mpi_features_and_polarity

__all__ = [
    'DataPreprocessor',
    'preprocess_data',
    'MPICalculator',
    'compute_mpi_features',
    'load_config',
    'get_mpi_features_and_polarity'
]

__version__ = '0.1.0'
