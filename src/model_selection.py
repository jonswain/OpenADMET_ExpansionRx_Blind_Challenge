"""Functions to select best performing model."""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger(__name__)


def train_model_selector(
    targets: list[str],
    cross_validation_predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, RandomForestClassifier]:
    cross_validation_predictions = _determine_best_model(
        targets, cross_validation_predictions
    )
    model_selectors: dict[str, RandomForestClassifier] = {}
    for target in targets:
        log.info("Training selection model for target: %s", target)
        selected_data = cross_validation_predictions[f"{target}_true"].dropna().index
        X = features.loc[selected_data]
        y = cross_validation_predictions[f"{target}_best_model"].loc[selected_data]
        log.info(
            "Proportion of classes: %s",
            y.value_counts(normalize=True).round(2).to_dict(),
        )
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X, y)
        model_selectors[target] = selector
    return model_selectors


def _determine_best_model(
    targets: list[str], cross_validation_predictions: pd.DataFrame
) -> pd.DataFrame:
    for target in targets:
        true_values = cross_validation_predictions[f"{target}_true"].dropna()
        model_cols = [
            col
            for col in cross_validation_predictions.columns
            if col.startswith(f"{target}_") and col != f"{target}_true"
        ]
        for model_col in model_cols:
            cross_validation_predictions[f"{model_col}_abs_error"] = abs(
                cross_validation_predictions[model_col] - true_values
            )
        performance_cols = cross_validation_predictions[
            [f"{col}_abs_error" for col in model_cols]
        ]
        # TODO: Silence error
        cross_validation_predictions[f"{target}_best_model"] = performance_cols.idxmin(
            axis=1, skipna=True
        ).str.replace("_abs_error", "")
        cross_validation_predictions[f"{target}_best_model"] = (
            cross_validation_predictions[f"{target}_best_model"].str.replace(
                f"{target}_", ""
            )
        )
    return cross_validation_predictions
