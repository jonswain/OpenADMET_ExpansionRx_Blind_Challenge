"""Functions to train Chemprop models for chemical data."""

import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .config import CROSS_VALIDATION_FOLDS


def train_cv_chemprop_models(
    data: pd.DataFrame, groups: pd.Series, uuid: str
) -> pd.DataFrame:
    """Train a Chemprop multitask models using cross-validation.

    Args:
        data (pd.DataFrame): A DataFrame containing the training data.
        groups (pd.Series): The groups to use for cross-validation.
        uuid (str): The UUID for the experiment.

    Returns:
        pd.DataFrame: A DataFrame containing cross-validation predictions.
    """
    Path(f"chemprop_data/{uuid}").mkdir(exist_ok=True, parents=True)
    Path(f"chemprop_models/{uuid}").mkdir(exist_ok=True, parents=True)
    cv = GroupKFold(n_splits=CROSS_VALIDATION_FOLDS)
    for fold, (train_idx, val_idx) in enumerate(cv.split(data, groups=groups)):
        data.drop("Molecule Name", axis=1).iloc[train_idx].to_csv(
            f"chemprop_data/{uuid}/train_fold_{fold}.csv"
        )
        data.drop("Molecule Name", axis=1).iloc[val_idx].to_csv(
            f"chemprop_data/{uuid}/val_fold_{fold}.csv"
        )

    for fold in tqdm(range(CROSS_VALIDATION_FOLDS)):
        subprocess.run(
            [
                "chemprop",
                "train",
                "--data-path",
                f"chemprop_data/{uuid}/train_fold_{fold}.csv",
                "--task-type",
                "regression",
                "--output-dir",
                f"chemprop_models/{uuid}/fold_{fold}",
                "--split-type",
                "scaffold_balanced",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                "chemprop",
                "predict",
                "--test-path",
                f"chemprop_data/{uuid}/val_fold_{fold}.csv",
                "--model-paths",
                f"chemprop_models/{uuid}/fold_{fold}",
                "--preds-path",
                f"chemprop_data/{uuid}/preds_fold_{fold}.csv",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    cross_val_preds = pd.concat(
        [
            pd.read_csv(f"chemprop_data/{uuid}/preds_fold_{fold}.csv")
            for fold in range(CROSS_VALIDATION_FOLDS)
        ],
        axis=0,
    ).set_index("SMILES")
    return cross_val_preds


def finalize_chemprop_model(data: pd.DataFrame, uuid: str) -> None:
    """Train a finalized Chemprop multitask model on all the training data.

    Args:
        data (pd.DataFrame): A DataFrame containing the training data.
        uuid (str): The UUID for the experiment.
    """
    Path(f"chemprop_data/{uuid}").mkdir(exist_ok=True, parents=True)
    Path(f"chemprop_models/{uuid}").mkdir(exist_ok=True, parents=True)
    data.drop("Molecule Name", axis=1).to_csv(f"chemprop_data/{uuid}/train.csv")
    subprocess.run(
        [
            "chemprop",
            "train",
            "--data-path",
            f"chemprop_data/{uuid}/train.csv",
            "--task-type",
            "regression",
            "--output-dir",
            f"chemprop_models/{uuid}/finalized_model",
            "--split-type",
            "scaffold_balanced",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
