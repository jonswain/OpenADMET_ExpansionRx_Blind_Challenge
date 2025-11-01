# OpenADMET and ExpansionRx Blind Challenge

Code for entry into the [OpenADMET + ExpansionRx Blind Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge).

## Model Design

Predictions are made using a ChemicalMetaRegressor. The ChemicalMetaRegressor is an ensemble approach for predicting multiple chemical properties (ADMET endpoints) that combines classical machine learning models with deep learning (Chemprop) and uses meta-learning for optimal model selection.

### Architecture & Workflow

#### Data Preprocessing

1. Standardizes and validates SMILES strings
2. Applies chemical preprocessing to clean the molecular data

#### Feature Engineering

1. Generates molecular fingerprints (ECFP with radius=2, 2048 bits)
2. Calculates 200+ RDKit 2D molecular descriptors
3. Creates Butina clusters for structure-based cross-validation splits

#### Multi-Model Training

1. Classical ML Models: Trains 9+ sklearn models (Linear, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, Gradient Boosting, Histogram GB, XGBoost, Stacking)
2. Deep Learning: Trains Chemprop multitask graph neural networks
3. Uses 5-fold group cross-validation (clustered by molecular similarity)

#### Meta-Learning Selection

1. Evaluates each model's cross-validation performance per prediction
2. Trains RandomForest classifiers to predict which model will perform best for each new molecule
3. Selection is based on molecular features and historical model performance patterns

#### Prediction Process

1. For new molecules: generates features → model selector predicts best model to use for predictions → uses selected model for final prediction
2. Each target gets its own model selector, allowing different models for different endpoints

### TODO

- Final models with hyperparameter tuning
- Look for most common classical ML model and try just that.
- Make into Python package
- Add trained parameter
- Pickle model for reuse
- Check CV folds are the same for all models

## Performances on Limited Test Set

### Second submission - MetaRegressor Trained on ECFP and RDKit2D Descriptors

| Endpoint     | MAE | R2 | Spearman R | Kendall's Tau |
| ---          | --- | --- | --- | --- |
| Overall      | 0.70 +/- 0.03 | 0.40 +/- 0.04 | 0.67 +/- 0.03 | 0.49 +/- 0.02 |
| LogD         | 0.46 +/- 0.02 | 0.58 +/- 0.04 | 0.79 +/- 0.02 | 0.62 +/- 0.01 |
| KSOL         | 0.46 +/- 0.01 | 0.46 +/- 0.02 | 0.60 +/- 0.02 | 0.42 +/- 0.02 |
| MLM CLint    | 0.40 +/- 0.01 | 0.29 +/- 0.04 | 0.55 +/- 0.03 | 0.39 +/- 0.02 |
| HLM CLint    | 0.35 +/- 0.01 | 0.21 +/- 0.06 | 0.51 +/- 0.04 | 0.36 +/- 0.03 |
| Caco2 Efflux | 0.39 +/- 0.01 | 0.05 +/- 0.04 | 0.65 +/- 0.03 | 0.48 +/- 0.02 |
| Caco2 A>B    | 0.29 +/- 0.01 | 0.15 +/- 0.05 | 0.56 +/- 0.03 | 0.39 +/- 0.02 |
| MPPB         | 0.23 +/- 0.01 | 0.49 +/- 0.04 | 0.71 +/- 0.03 | 0.51 +/- 0.03 |
| MBPB         | 0.17 +/- 0.01 | 0.71 +/- 0.03 | 0.82 +/- 0.03 | 0.64 +/- 0.02 |
| MGMB         | 0.18 +/- 0.01 | 0.65 +/- 0.05 | 0.80 +/- 0.03 | 0.62 +/- 0.03 |

### Initial submission - Chemprop Multitask Predictions

| Endpoint     | MAE | R2 | Spearman R | Kendall's Tau |
| ---          | --- | --- | --- | --- |
| Overall      | 0.79 +/- 0.03 | 0.21 +/- 0.06 | 0.63 +/- 0.03 | 0.47 +/- 0.02 |
| LogD         | 0.49 +/- 0.01 | 0.62 +/- 0.02 | 0.81 +/- 0.01 | 0.63 +/- 0.01 |
| KSOL         | 0.43 +/- 0.01 | 0.48 +/- 0.03 | 0.64 +/- 0.02 | 0.46 +/- 0.02 |
| MLM CLint    | 0.42 +/- 0.01 | 0.17 +/- 0.05 | 0.50 +/- 0.03 | 0.35 +/- 0.02 |
| HLM CLint    | 0.38 +/- 0.01 | 0.10 +/- 0.08 | 0.49 +/- 0.04 | 0.35 +/- 0.03 |
| Caco2 Efflux | 0.47 +/- 0.01 | -0.38 +/- 0.06 | 0.54 +/- 0.03 | 0.39 +/- 0.02 |
| Caco2 A>B    | 0.43 +/- 0.01 | -0.81 +/- 0.11 | 0.46 +/- 0.03 | 0.31 +/- 0.02 |
| MPPB         | 0.21 +/- 0.01 | 0.55 +/- 0.05 | 0.74 +/- 0.03 | 0.56 +/- 0.03 |
| MBPB         | 0.21 +/- 0.01 | 0.57 +/- 0.04 | 0.77 +/- 0.03 | 0.57 +/- 0.03 |
| MGMB         | 0.18 +/- 0.01 | 0.60 +/- 0.07 | 0.76 +/- 0.04 | 0.60 +/- 0.04 |

## Local Development

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

### Create Environment

The following commands will setup an environment where you can run and test the application locally:

```bash
git clone git@github.com:jonswain/OpenADMET_ExpansionRx_Blind_Challenge
cd OpenADMET_ExpansionRx_Blind_Challenge
conda env create -f environment.yml
conda activate OpenADMET_challenge
code .
```
