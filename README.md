# Urban Prediction of Mazziotta–Pareto Index (MPI)

This repository will host the end-to-end workflow to:

- Construct the Mazziotta–Pareto Index (MPI) using rich urban features in areas where detailed data is available, and
- Learn a simpler predictive model that uses widely available variables (e.g., census/household survey aggregates) to estimate the MPI in other areas with limited data.

In short: we build the MPI where we have many signals, then predict that index elsewhere using leaner inputs.

## Reference
Developing the urban comfort index: Advancing liveability analytics with a multidimensional approach and explainable artificial intelligence (Binyu Lei, Pengyuan Liu, Xiucheng Liang, Yingwei Yan, Filip Biljecki)

https://doi.org/10.1016/j.scs.2024.106121

## Why MPI?

The Mazziotta–Pareto approach is a composite indicator methodology that aggregates multiple standardized indicators into a single score while penalizing imbalance among dimensions. It’s useful for producing robust area-level indices that reflect multiple aspects of urban well-being or deprivation without allowing one dimension to completely compensate for another.


## Project objectives

1. Define the indicator set and polarity (beneficial vs. non-beneficial) for an urban MPI.
2. Compute the MPI in those regions using a transparent, reproducible pipeline.
3. Train a supervised model that maps simpler, widely available variables (primarily census aggregates) to the MPI.
4. Predict the MPI for target regions with scarce data and validate transferability.


## Deliverables
- Reproducible script for feature engineering, MPI computation, model training, and inference.
- Artifacts: processed datasets, trained models, and predicted MPI geo-packages as appropriate.


## Data

- Urban feature sources:
    - Census Brazil (IBGE) - income, households, and population.
    - Infrastructure - highways, roads, streets, and slope of the terrain.
    - POI - supermarkets, schools, prision, cemetery, and parks.

- Predictor set for transfer (examples):
  - Census/household survey aggregates (income and demographics)


## Repository structure (planned)


```
urban-prediction-mpi-index/
  data/               # raw/, interim/, processed/ (not tracked or with LFS)
  notebooks/          # EDA, feature engineering, MPI, modeling, inference
  src/                # reusable processing and modeling code
  configs/            # indicator lists, polarity, parameters
  README.md
```