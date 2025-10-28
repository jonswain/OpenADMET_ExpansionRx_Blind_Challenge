"""Functions for making predictions using trained models."""

import logging

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from .features import generate_features

log = logging.getLogger(__name__)


def predict_on_smiles(
    smiles: list[str] | pd.Series,
    selection_models: dict[str, RandomForestClassifier],
    classical_models: dict[str, dict[str, BaseEstimator]],
) -> pd.DataFrame:
    test_features = generate_features(smiles)
    prediction_df = pd.DataFrame(index=smiles)
    for target, selector in selection_models.items():
        log.info(f"Making predictions for target: {target}")
        prediction_df[f"{target}_model"] = selector.predict(test_features)
        for model_name in prediction_df[f"{target}_model"].unique():
            log.info(f"Using model: {model_name}")
            model = classical_models[target][model_name]
            selected_smiles = prediction_df.index[
                prediction_df[f"{target}_model"] == model_name
            ]
            if not selected_smiles.empty:
                selected_features = test_features.loc[selected_smiles]
                preds = model.predict(selected_features)
                prediction_df.loc[selected_smiles, target] = preds
    return prediction_df[
        [col for col in prediction_df.columns if not col.endswith("_model")]
    ]
