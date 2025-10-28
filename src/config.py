"""Configuration file for model parameters and settings."""

from typing import Callable

from optuna import Trial
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from xgboost import XGBRegressor

CROSS_VALIDATION_FOLDS = 5

SKLEARN_MODELS = [
    LinearRegression(n_jobs=-1),
    Lasso(),
    Ridge(),
    ElasticNet(),
    ExtraTreesRegressor(n_jobs=-1, random_state=42),
    GradientBoostingRegressor(random_state=42),
    HistGradientBoostingRegressor(random_state=42),
    RandomForestRegressor(n_jobs=-1, random_state=42),
    XGBRegressor(n_jobs=-1, random_state=42),
    StackingRegressor(
        estimators=[
            ("et", ExtraTreesRegressor(n_jobs=-1, random_state=42)),
            ("lr", Ridge(alpha=1.0)),
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1,
    ),
]


def _sample_lasso_params(trial: Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
    }


def _sample_ridge_params(trial: Trial) -> dict:
    """Search space for Ridge Regressor."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-6, 100.0, log=True),
    }


def _sample_elasticnet_params(trial: Trial) -> dict:
    """Search space for ElasticNet Regressor."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 1e-4, 1.0, log=True),
    }


def _sample_et_params(trial: Trial) -> dict:
    """Search space for ExtraTreesRegressor."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 20, 80, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.3, 0.8, step=0.1),
        "n_jobs": -1,
    }


def _sample_gbr_params(trial: Trial) -> dict:
    """Search space for GradientBoostingRegressor."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
    }


def _sample_hgb_params(trial: Trial) -> dict:
    """Search space for HistGradientBoostingRegressor (faster, fewer params)."""
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }


def _sample_rf_params(trial: Trial) -> dict:
    """Search space for RandomForestRegressor."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=100),
        "max_depth": trial.suggest_int("max_depth", 10, 50, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.6, 1.0, step=0.1),
        "n_jobs": -1,
    }


def _sample_xgb_params(trial: Trial) -> dict:
    """Search space for XGBRegressor."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-2, 1e2, log=True
        ),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0, step=0.05
        ),
        "n_jobs": -1,
        "tree_method": "hist",
    }


def _sample_stacking_params(trial: Trial) -> dict:
    """Search space for StackingRegressor (ExtraTrees + Ridge + Ridge Meta)."""
    et_params = {
        "et__n_estimators": trial.suggest_int("et__n_estimators", 100, 500, step=100),
        "et__max_depth": trial.suggest_int("et__max_depth", 15, 30),
        "et__min_samples_leaf": trial.suggest_int("et__min_samples_leaf", 1, 5),
        "et__max_features": trial.suggest_float("et__max_features", 0.5, 1.0, step=0.1),
    }
    lr_params = {
        "lr__alpha": trial.suggest_float("lr__alpha", 1e-3, 10.0, log=True),
    }
    final_estimator_params = {
        "final_estimator__alpha": trial.suggest_float(
            "final_estimator__alpha", 0.1, 100.0, log=True
        ),
    }
    stacking_params = {
        "cv": trial.suggest_int("cv", 3, 5),
    }
    params = {}
    params.update(et_params)
    params.update(lr_params)
    params.update(final_estimator_params)
    params.update(stacking_params)
    return params


HYPERPARAMETER_SEARCH_SPACE: dict[str, Callable[[Trial], dict]] = {
    Lasso.__name__: _sample_lasso_params,
    Ridge.__name__: _sample_ridge_params,
    ElasticNet.__name__: _sample_elasticnet_params,
    RandomForestRegressor.__name__: _sample_rf_params,
    ExtraTreesRegressor.__name__: _sample_et_params,
    GradientBoostingRegressor.__name__: _sample_gbr_params,
    HistGradientBoostingRegressor.__name__: _sample_hgb_params,
    XGBRegressor.__name__: _sample_xgb_params,
    StackingRegressor.__name__: _sample_stacking_params,
}
