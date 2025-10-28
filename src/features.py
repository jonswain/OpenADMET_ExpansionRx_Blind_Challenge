"""Functions for feature engineering and data preprocessing."""

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from tqdm import tqdm

log = logging.getLogger(__name__)


def generate_features(
    smiles_list: list | pd.Series, radius: int = 2, fpSize: int = 2048
) -> pd.DataFrame:
    """_summary_

    Args:
        smiles_list (list | pd.Series): _description_
        radius (int, optional): _description_. Defaults to 2.
        fpSize (int, optional): _description_. Defaults to 2048.

    Returns:
        pd.DataFrame: _description_
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


def _generate_rdkit_descriptors(mol, missingVal=None) -> dict:
    """_summary_

    Args:
        mol (_type_): _description_
        missingVal (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        res[nm] = val
    return res
