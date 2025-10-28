"""Functions to train Chemprop models for chemical data."""

import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm

from .config import CROSS_VALIDATION_FOLDS


def train_cv_chemprop_models(
    data: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """Train a Chemprop multitask model on the data."""
    Path("chemprop_data").mkdir(exist_ok=True)
    Path("chemprop_models").mkdir(exist_ok=True)
    cv = KFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(cv.split(data)):
        data.drop("Molecule Name", axis=1).iloc[train_idx].to_csv(
            f"chemprop_data/train_fold_{fold}.csv", index=False
        )
        data.drop("Molecule Name", axis=1).iloc[val_idx].to_csv(
            f"chemprop_data/val_fold_{fold}.csv", index=False
        )

    for fold in tqdm(range(CROSS_VALIDATION_FOLDS)):
        subprocess.run(
            [
                "chemprop",
                "train",
                "--data-path",
                f"chemprop_data/train_fold_{fold}.csv",
                "--task-type",
                "regression",
                "--output-dir",
                f"chemprop_models/fold_{fold}",
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
                f"chemprop_data/val_fold_{fold}.csv",
                "--model-paths",
                f"chemprop_models/fold_{fold}",
                "--preds-path",
                f"chemprop_data/preds_fold_{fold}.csv",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    cross_val_preds = pd.concat(
        [
            pd.read_csv(f"chemprop_data/preds_fold_{fold}.csv")
            for fold in range(CROSS_VALIDATION_FOLDS)
        ],
        axis=0,
    ).set_index(smiles_col)
    return cross_val_preds


def finalize_chemprop_model(data: pd.DataFrame) -> pd.DataFrame:
    """Train a Chemprop multitask model on the data."""
    Path("chemprop_data").mkdir(exist_ok=True)
    Path("chemprop_models").mkdir(exist_ok=True)
    data.drop("Molecule Name", axis=1).to_csv(f"chemprop_data/train.csv", index=False)
    subprocess.run(
        [
            "chemprop",
            "train",
            "--data-path",
            f"chemprop_data/train.csv",
            "--task-type",
            "regression",
            "--output-dir",
            f"chemprop_models/finalized_model",
            "--split-type",
            "scaffold_balanced",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
