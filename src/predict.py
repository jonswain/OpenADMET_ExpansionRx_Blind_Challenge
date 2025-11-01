"""Functions for making predictions using trained models."""

import logging
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import generate_features

log = logging.getLogger(__name__)


def predict_on_smiles(
    smiles: list[str] | pd.Series,
    selection_models: dict[str, RandomForestClassifier],
    classical_models: dict[str, dict[str, Pipeline]],
    uuid: str,
    return_model_choices: bool = False,
) -> pd.DataFrame:
    """Make predictions for each SMILES in the test set using the best model.

    Args:
        smiles (list[str] | pd.Series): The SMILES to make predictions for.
        selection_models (dict[str, RandomForestClassifier]): The models to use to
            select the best performing model for each target.
        classical_models (dict[str, dict[str, Pipeline]]): A dictionary mapping target
            names to their trained classical models.
        uuid (str): The unique identifier for the model instance.
        return_model_choices (bool): Whether to return the model choices alongside
            the predictions. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions for each target.
    """
    test_features = generate_features(smiles)
    prediction_df = pd.DataFrame(index=smiles)
    for target, selector in selection_models.items():
        log.info(f"Making model selection predictions for target: {target}")
        prediction_df[f"{target}_model"] = selector.predict(test_features)
    if "Chemprop" in prediction_df.values:
        log.info("Making Chemprop predictions")
        chemprop_preds = _make_chemprop_predictions(smiles, uuid)
    for target, models in classical_models.items():
        log.info(f"Making predictions for target: {target}")
        for model_name in prediction_df[f"{target}_model"].unique():
            if model_name == "Chemprop":
                log.info("Using model: Chemprop")
                selected_smiles = prediction_df[
                    prediction_df[f"{target}_model"] == model_name
                ].index
                prediction_df.loc[selected_smiles, target] = chemprop_preds.loc[
                    selected_smiles, target
                ]
            else:
                log.info(f"Using model: {model_name}")
                model = models[model_name]
                selected_smiles = prediction_df[
                    prediction_df[f"{target}_model"] == model_name
                ].index
                selected_features = test_features.loc[selected_smiles]
                preds = model.predict(selected_features)
                prediction_df.loc[selected_smiles, target] = preds
    if return_model_choices:
        return prediction_df
    return prediction_df[
        [col for col in prediction_df.columns if not col.endswith("_model")]
    ]


def _make_chemprop_predictions(
    smiles: list[str] | pd.Series, uuid: str
) -> pd.DataFrame:
    """Make Chemprop predictions on a list of SMILES strings.

    Args:
        smiles (list[str] | pd.Series): The SMILES strings to make predictions for.
        uuid (str): The unique identifier for the model instance.

    Returns:
        pd.DataFrame: A DataFrame containing the Chemprop predictions.
    """
    Path(f"chemprop_data/{uuid}").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SMILES": smiles}).to_csv(
        f"chemprop_data/{uuid}/test.csv", index=False
    )
    subprocess.run(
        [
            "chemprop",
            "predict",
            "--test-path",
            f"chemprop_data/{uuid}/test.csv",
            "--model-paths",
            f"chemprop_models/{uuid}",
            "--preds-path",
            f"chemprop_data/{uuid}/preds.csv",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return pd.read_csv(f"chemprop_data/{uuid}/preds.csv").set_index("SMILES")
