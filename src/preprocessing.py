"""Functions for chemical data preprocessing and standardization."""

import logging

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


class ChemicalStandardizer:
    """Standardize chemical structures for machine learning."""

    def __init__(self):
        self.salt_remover = SaltRemover.SaltRemover()
        self.normalizer = rdMolStandardize.Normalizer()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def standardize_smiles(self, smiles: str) -> str | None:
        """Standardize a SMILES string.

        Args:
            smiles (str): Input SMILES string.

        Returns:
            Optional[str]: Standardized SMILES or None if invalid.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = self.salt_remover.StripMol(mol)
            mol = self.normalizer.normalize(mol)
            mol = self.tautomer_enumerator.Canonicalize(mol)
            mol = Chem.AddHs(mol)
            mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(mol, canonical=True)

        except Exception as e:
            log.warning(f"Failed to standardize SMILES {smiles}: {e}")
            return None

    def filter_molecules(
        self, df: pd.DataFrame, feature_cols: list[str], smiles_col: str = "SMILES"
    ) -> pd.DataFrame:
        """Filter molecules based on chemical validity and drug-like properties.

        Args:
            df (pd.DataFrame): DataFrame containing SMILES.
            feature_cols (list[str]): List of feature column names.
            smiles_col (str): Column name containing SMILES. Defaults to "SMILES".

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        df = df.copy()
        df[f"{smiles_col}_standardized"] = df[smiles_col].apply(self.standardize_smiles)
        initial_count = len(df)
        df = df.dropna(subset=[f"{smiles_col}_standardized"])
        log.info(f"Removed {initial_count - len(df)} invalid molecules")
        df = self._apply_molecular_filters(df, f"{smiles_col}_standardized")
        duplicate_mask = df.duplicated(
            subset=[f"{smiles_col}_standardized"], keep=False
        )
        duplicate_smiles = df.loc[duplicate_mask, f"{smiles_col}_standardized"].unique()
        if len(duplicate_smiles) > 0:
            log.info(f"Averaging {len(duplicate_smiles)} duplicate molecules")
            for dup_smiles in duplicate_smiles:
                dup_indices = df.index[
                    df[f"{smiles_col}_standardized"] == dup_smiles
                ].tolist()
                if len(dup_indices) > 1:
                    averaged_features = df.loc[dup_indices, feature_cols].mean()
                    df.loc[dup_indices[0], feature_cols] = averaged_features
                    df = df.drop(index=dup_indices[1:])
        df[smiles_col] = df[f"{smiles_col}_standardized"]
        df = df.drop(columns=[f"{smiles_col}_standardized"])
        return df

    def _apply_molecular_filters(
        self, df: pd.DataFrame, smiles_col: str
    ) -> pd.DataFrame:
        """Apply drug-like molecular filters.

        Args:
            df (pd.DataFrame): DataFrame with SMILES.
            smiles_col (str): SMILES column name.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """

        def calculate_properties(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                return {
                    "mw": Descriptors.MolWt(mol),
                    "logp": Descriptors.MolLogP(mol),
                    "hbd": Descriptors.NumHDonors(mol),
                    "hba": Descriptors.NumHAcceptors(mol),
                    "tpsa": Descriptors.TPSA(mol),
                    "rotbonds": Descriptors.NumRotatableBonds(mol),
                    "heavy_atoms": mol.GetNumHeavyAtoms(),
                }
            except Exception:
                return None

        props = df[smiles_col].apply(calculate_properties)
        props_df = pd.DataFrame(props.tolist())
        filters = (
            (props_df["mw"] >= 50)
            & (props_df["mw"] <= 1000)
            & (props_df["logp"] >= -3)
            & (props_df["logp"] <= 7)
            & (props_df["hbd"] <= 10)
            & (props_df["hba"] <= 15)
            & (props_df["tpsa"] <= 200)
            & (props_df["rotbonds"] <= 15)
            & (props_df["heavy_atoms"] >= 3)
            & (props_df["heavy_atoms"] <= 100)
        )
        initial_count = len(df)
        df = df[filters.fillna(False)]
        log.info(f"Molecular filters removed {initial_count - len(df)} compounds")

        return df


def preprocess_chemical_data(
    df: pd.DataFrame, feature_cols: list[str], smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """Preprocess chemical data with standardization and filtering.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list[str]): List of feature column names.
        smiles_col (str): Column containing SMILES strings. Defaults to "SMILES".

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    standardizer = ChemicalStandardizer()
    return standardizer.filter_molecules(df, feature_cols, smiles_col)
