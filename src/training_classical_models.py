"""Functions for training classical machine learning models for chemical data."""

import logging
from time import time

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CROSS_VALIDATION_FOLDS, HYPERPARAMETER_SEARCH_SPACE, SKLEARN_MODELS

log = logging.getLogger(__name__)
logging.getLogger("optuna").setLevel(logging.WARNING)


def train_classical_models(
    features: pd.DataFrame,
    data: pd.DataFrame,
    smiles_col: str,
    target_cols: list[str],
    clusters: pd.Series,
    n_keep: int = 1,
) -> dict[str, dict[str, Pipeline]]:
    """Train classifcal ML models for each target.

    Args:
        features (pd.DataFrame): The feature data to use for training.
        data (pd.DataFrame): The target data and SMILES identifiers to use for training.
        smiles_col (str): The column name for the SMILES identifiers.
        target_cols (list[str]): The column names for the targets.
        clusters (pd.Series): The groups to use for cross-validation.
        n_keep (int, optional): Number of best models to keep. Defaults to 1.

    Returns:
        dict[str, dict[str, Pipeline]]: A dictionary containing the trained models.
    """
    trained_models: dict[str, dict[str, Pipeline]] = {}
    for target in target_cols:
        log.info("Training classical models for target: %s", target)
        selected_data = data[[smiles_col, target]].dropna(subset=[target])
        log.info("Number of training samples %d", len(selected_data))
        X = features.loc[selected_data[smiles_col]]
        y = selected_data[target].values
        groups = clusters.loc[selected_data[smiles_col]]
        best_models = _train_classical_model(X, y, groups=groups, n_keep=n_keep)
        trained_models[target] = {}
        for model in best_models:
            trained_models[target][model["regressor"].__class__.__name__] = model
    return trained_models


def _train_classical_model(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_keep: int = 1
) -> list[Pipeline]:
    """Train the best classical regression model on the provided data.

    Args:
        X (pd.DataFrame): The features to use for training.
        y (pd.Series): The labels to use for training.
        groups (pd.Series): The groups to use for cross-validation.
        n_keep (int, optional): Number of best models to keep. Defaults to 1.

    Returns:
        list[Pipeline]: The trained models.
    """
    models = _regression_selection_by_cv(X, y, groups)
    best_models = sorted(models, key=lambda model: model.performance)[:n_keep]
    for model in best_models:
        if model["regressor"].__class__.__name__ not in HYPERPARAMETER_SEARCH_SPACE:
            best_params = {}
        else:
            log.info(
                "Performing hyperparameter tuning for %s",
                model["regressor"].__class__.__name__,
            )
            best_params = _hyperparameter_tuning_by_cv(model, X, y, groups)
            log.info(
                "Best hyperparameters for %s: %s",
                model["regressor"].__class__.__name__,
                best_params,
            )
        model["regressor"].set_params(**best_params)
        model.fit(X, y)
    return best_models


def _regression_selection_by_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> list[Pipeline]:
    """Perform cross-validation on the regression models to get the best algorithm.

    Args:
        X (pd.DataFrame): The features to use for training.
        y (pd.Series): The labels to use for training.
        groups (pd.Series): The groups to use for cross-validation.

    Returns:
        list[Pipeline]: The trained models with cross-validation scores.
    """
    scores = []
    for reg in SKLEARN_MODELS:
        cv = GroupKFold(n_splits=CROSS_VALIDATION_FOLDS)
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


def objective_wrapper(
    trial: optuna.Trial,
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
) -> float:
    """Determine the objective for hyperparameter tuning.

    Args:
        trial (optuna.Trial): The current optimization trial.
        model (Pipeline): The model to tune.
        X (pd.DataFrame): The features to use for training.
        y (np.ndarray): The target to use for training.
        groups (pd.Series): The groups to use for cross-validation.
                                   Defaults to None.

    Raises:
        ValueError: If the model is not found in the hyperparameter search space.

    Returns:
        float: The mean absolute error for the model with the given hyperparameters.
    """
    model_name = model["regressor"].__class__.__name__
    if model_name not in HYPERPARAMETER_SEARCH_SPACE:
        raise ValueError(
            f"Model {model_name} not found in HYPERPARAMETER_SEARCH_SPACE."
        )
    params = HYPERPARAMETER_SEARCH_SPACE[model_name](trial)
    cv = GroupKFold(n_splits=CROSS_VALIDATION_FOLDS)
    cv_splits = cv.split(X, y, groups)
    tuned_model = clone(model["regressor"]).set_params(**params)
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("regressor", tuned_model)])
    mae_list: list[float] = []
    for i, (train_index, val_index) in enumerate(cv_splits):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        mae_list.append(mae)

        current_mean_mae = float(np.mean(mae_list))
        trial.report(current_mean_mae, i)
        if trial.should_prune():
            raise TrialPruned()

    return float(np.mean(mae_list))


def _hyperparameter_tuning_by_cv(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> dict:
    """Perform hyperparameter tuning on the provided model using cross-validation.

    Args:
        model (BaseEstimator): The model to tune.
        X (pd.DataFrame): The features to use for training.
        y (pd.Series): The labels to use for training.
        groups (pd.Series): The groups to use for cross-validation.

    Returns:
        dict: The best hyperparameters found.
    """
    objective_for_optuna = lambda trial: objective_wrapper(trial, model, X, y, groups)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective_for_optuna, n_trials=100, show_progress_bar=True)
    return study.best_params
