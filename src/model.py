"""Model definition for chemical property prediction."""

import logging
from dataclasses import dataclass, field

import pandas as pd

from .features import calculate_butina_clusters, generate_features
from .model_selection import train_model_selector
from .predict import predict_on_smiles
from .training_chemprop_models import finalize_chemprop_model, train_cv_chemprop_models
from .training_classical_models import train_classical_models

log = logging.getLogger(__name__)


@dataclass
class ChemicalMetaRegressor:
    """Meta-regressor for predicting chemical properties.

    Attributes:
        smiles_col (str): The name of the column containing SMILES strings.
        target_cols (list[str]): A list of target column names.
        training_data (pd.DataFrame): The training data containing SMILES and target values.
        clusters (pd.Series): The Butina clusters for the training data.
        training_features (pd.DataFrame): The generated features for the training data.
        classical_models (dict): A dictionary of trained classical models.
        cross_val_preds (pd.DataFrame): A DataFrame containing cross-validation predictions.
        model_selectors (dict): A dictionary of trained model selectors.
    """

    smiles_col: str = "SMILES"
    target_cols: list[str] = field(default_factory=list)
    training_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    clusters: pd.Series = field(default_factory=pd.Series)
    training_features: pd.DataFrame = field(init=False)
    classical_models: dict = field(default_factory=dict)
    cross_val_preds: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_selectors: dict = field(default_factory=dict)

    def __post_init__(self):
        log.info("Generating features for training data")
        self.training_features = generate_features(self.training_data[self.smiles_col])
        self.clusters = calculate_butina_clusters(self.training_features)
        self.cross_val_preds = pd.DataFrame()
        self.cross_val_preds[self.smiles_col] = self.training_data[self.smiles_col]
        for target in self.target_cols:
            self.cross_val_preds[f"{target}_true"] = self.training_data[target]
        self.cross_val_preds.set_index(self.smiles_col, inplace=True)

    def _train_classical_models(
        self, n_keep: int = 1, tune_hyperparameters: bool = True
    ):
        log.info("Training classical models")
        self.classical_models = train_classical_models(
            self.training_features,
            self.training_data,
            self.smiles_col,
            self.target_cols,
            self.clusters,
            n_keep=n_keep,
            tune_hyperparameters=tune_hyperparameters,
        )
        for target, models in self.classical_models.items():
            for model_name, model in models.items():
                self.cross_val_preds.loc[model.smiles, f"{target}_{model_name}"] = (
                    model.cross_val_preds
                )

    def _train_chemprop_model(self):
        """Train a Chemprop model on the training data."""
        log.info("Training Chemprop models")
        chemprop_cross_val_preds = train_cv_chemprop_models(
            self.training_data, self.clusters, self.smiles_col
        )
        chemprop_cross_val_preds.columns = [
            f"{col}_Chemprop" for col in chemprop_cross_val_preds.columns
        ]
        self.cross_val_preds = self.cross_val_preds.merge(
            chemprop_cross_val_preds,
            left_index=True,
            right_index=True,
        )
        finalize_chemprop_model(self.training_data)

    def _train_model_selector(self):
        """Train a model selector to choose the best model for each prediction."""
        log.info("Training model selectors")
        self.model_selectors = train_model_selector(
            self.target_cols,
            self.cross_val_preds,
            self.training_features,
        )

    def train_models(
        self, n_keep_classical: int = 1, tune_hyperparameters: bool = True
    ):
        """Train all models (classical and Chemprop) on the training data."""
        self._train_classical_models(
            n_keep=n_keep_classical, tune_hyperparameters=tune_hyperparameters
        )
        self._train_chemprop_model()
        self._train_model_selector()

    def predict(self, smiles: list[str] | pd.Series) -> pd.DataFrame:
        """Make predictions on new SMILES strings."""
        log.info("Making predictions on new %s SMILES strings", len(smiles))
        return predict_on_smiles(
            smiles,
            self.model_selectors,
            self.classical_models,
        )
