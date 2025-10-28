"""Functions for training classical machine learning models for chemical data."""

import logging
from time import time

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def train_classical_models(
    features: pd.DataFrame,
    data: pd.DataFrame,
    smiles_col: str,
    target_cols: list[str],
    n_keep: int = 1,
) -> dict[str, dict[str, BaseEstimator]]:
    trained_models: dict[str, dict[str, BaseEstimator]] = {}
    for target in target_cols:
        log.info("Training classical models for target: %s", target)
        selected_data = data[[smiles_col, target]].dropna(subset=[target])
        log.info("Number of training samples %d", len(selected_data))
        X = features.loc[selected_data[smiles_col]]
        y = selected_data[target].values
        # TODO: Add support for grouping
        best_models = _train_classical_model(X, y, groups=None, n_keep=n_keep)
        trained_models[target] = {}
        for model in best_models:
            trained_models[target][model["regressor"].__class__.__name__] = model
    return trained_models


def _train_classical_model(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None, n_keep: int = 1
) -> BaseEstimator:
    """Train the best classical regression model on the provided data.

    Args:
        X (pd.DataFrame): The features to use for training.
        y (pd.Series): The labels to use for training.
        groups (pd.Series | None): The groups to use for cross-validation.

    Returns:
        BaseEstimator: The trained model.
    """
    models = _regression_selection_by_cv(X, y, groups)
    best_models = sorted(models, key=lambda model: model.performance)[:n_keep]
    for model in best_models:
        # TODO: Hyperparameter tuning can be added here
        best_params: dict = {}
        model.set_params(**best_params)
        model.fit(X, y)
    return best_models


def _regression_selection_by_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> list[BaseEstimator]:
    """Perform cross-validation on the regression models to get the best algorithm.

    Args:
        X (pd.DataFrame): The features to use for training.
        y (pd.Series): The labels to use for training.
        groups (pd.Series): The groups to use for cross-validation.

    Returns:
        list[BaseEstimator]: The trained models with cross-validation scores.
    """
    scores = []
    for reg in [
        LinearRegression(n_jobs=-1),
        Lasso(),
        Ridge(),
        ElasticNet(),
        ExtraTreesRegressor(n_jobs=-1, random_state=42),
        GradientBoostingRegressor(random_state=42),
        HistGradientBoostingRegressor(random_state=42),
        RandomForestRegressor(n_jobs=-1, random_state=42),
        StackingRegressor(
            estimators=[
                ("et", ExtraTreesRegressor(n_jobs=-1, random_state=42)),
                ("lr", Ridge(alpha=1.0)),
            ],
            final_estimator=Ridge(alpha=1.0),
            n_jobs=-1,
        ),
    ]:
        if groups is not None:
            cv = GroupKFold(n_splits=5)
        else:
            cv = KFold(n_splits=5, random_state=42, shuffle=True)
        start = time()
        scaler = StandardScaler()
        pipe = Pipeline(steps=[("scaler", scaler), ("regressor", reg)])
        predictions = cross_val_predict(estimator=pipe, X=X, y=y, cv=cv, groups=groups)
        mae = mean_absolute_error(y, predictions)
        log.info(
            "%s %s: %.2f (in %.2fs)",
            reg.__class__.__name__,
            "MAE",
            mae,
            time() - start,
        )
        pipe.smiles = X.index.tolist()
        pipe.cross_val_preds = predictions
        pipe.performance = mae
        scores.append(pipe)
    return scores
