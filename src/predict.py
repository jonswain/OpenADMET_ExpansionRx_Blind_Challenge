"""Functions for making predictions using trained models."""

import logging
import subprocess
from pathlib import Path

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
    use_chemprop = False
    test_features = generate_features(smiles)
    prediction_df = pd.DataFrame(index=smiles)
    for target, selector in selection_models.items():
        log.info(f"Making predictions for target: {target}")
        prediction_df[f"{target}_model"] = selector.predict(test_features)
        for model_name in prediction_df[f"{target}_model"].unique():
            if model_name == "Chemprop":
                use_chemprop = True
                continue
            log.info(f"Using model: {model_name}")
            model = classical_models[target][model_name]
            selected_smiles = prediction_df.index[
                prediction_df[f"{target}_model"] == model_name
            ]
            if not selected_smiles.empty:
                selected_features = test_features.loc[selected_smiles]
                preds = model.predict(selected_features)
                prediction_df.loc[selected_smiles, target] = preds
    if use_chemprop:
        chemprop_preds = _make_chemprop_predictions(smiles, smiles_col="SMILES")
    # Where prediction_df is na, fill in from chemprop_preds
        for target in selection_models.keys():
            missing_smiles = prediction_df.index[prediction_df[target].isna()]
            if not missing_smiles.empty:
                preds = chemprop_preds.loc[missing_smiles, target]
                prediction_df.loc[missing_smiles, target] = preds
        print(chemprop_preds)
    
    
    return prediction_df[
        [col for col in prediction_df.columns if not col.endswith("_model")]
    ]


def _make_chemprop_predictions(
    smiles: list[str] | pd.Series, smiles_col: str
) -> pd.DataFrame:
    """Make Chemprop predictions on a list of SMILES strings."""
    log.info("Making Chemprop predictions")
    Path("chemprop_data/test.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({smiles_col: smiles}).to_csv("chemprop_data/test.csv", index=False)
    subprocess.run(
        [
            "chemprop",
            "predict",
            "--test-path",
            f"chemprop_data/test.csv",
            "--model-paths",
            "chemprop_models/finalized_model",
            "--preds-path",
            f"chemprop_data/preds.csv",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return pd.read_csv("chemprop_data/preds.csv").set_index(smiles_col)
