"""Functions for feature engineering and data preprocessing."""

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdFingerprintGenerator
from tqdm import tqdm

log = logging.getLogger(__name__)


def generate_features(
    smiles_list: list | pd.Series, radius: int = 2, fpSize: int = 2048
) -> pd.DataFrame:
    """Generate molecular features from a list of SMILES strings.

    Args:
        smiles_list (list | pd.Series): A list or pandas Series of SMILES strings.
        radius (int, optional): The radius for the Morgan fingerprint. Defaults to 2.
        fpSize (int, optional): The length to fold the fingerprint to. Defaults to 2048.

    Returns:
        pd.DataFrame: A DataFrame containing the generated features.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    mols = [
        Chem.MolFromSmiles(s) for s in tqdm(smiles_list, desc="Generating molecules")
    ]
    fps = [
        mfpgen.GetCountFingerprintAsNumPy(mol)
        for mol in tqdm(mols, desc="Generating fingerprints")
    ]
    fps_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(fpSize)])
    descriptors = [
        _generate_rdkit_descriptors(mol)
        for mol in tqdm(mols, desc="Generating descriptors")
    ]
    desc_df = pd.DataFrame(descriptors)
    combined_df = pd.concat([fps_df, desc_df], axis=1)
    combined_df.index = smiles_list
    return combined_df


def _generate_rdkit_descriptors(mol: Mol, missingVal: None | float = None) -> dict:
    """Generate all RDKit 2D descriptors for a given molecule.

    Args:
        mol (Mol): The RDKit molecule object.
        missingVal (None | float, optional): The value to use for missing descriptors.
                                             Defaults to None.

    Returns:
        dict: A dictionary of descriptor names and their values.
    """
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        res[nm] = val
    return res
