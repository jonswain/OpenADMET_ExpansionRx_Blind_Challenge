"""Functions for feature engineering and data preprocessing."""

import logging

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Mol, rdFingerprintGenerator
from rdkit.DataStructs import IntSparseIntVect
from rdkit.ML.Cluster import Butina
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


def calculate_butina_clusters(
    features_df: pd.DataFrame, cutoff: float = 0.35, fpSize: int = 2048
) -> pd.Series:
    """Calculate Butina clusters for grouping during cross-validation.

    Args:
        features_df (pd.DataFrame): The DataFrame containing molecular fingerprints.
        cutoff (float, optional): The Tanimoto distance cutoff for clustering. Defaults to 0.35.
        fpSize (int, optional): The size of the fingerprint. Defaults to 2048.

    Returns:
        pd.Series: A Series containing the cluster labels for each molecule.
    """
    fps = features_df[[f"fp_{i}" for i in range(fpSize)]]
    dists = []
    nfps = len(fps)
    fps_list = fps.apply(_series_to_bit_vect, axis=1).tolist()
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps_list[i], fps_list[:i])
        dists.extend([1 - x for x in sims])
    mol_clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = [0] * nfps
    for idx, cluster in enumerate(mol_clusters, 1):
        for member in cluster:
            cluster_id_list[member] = idx
    clusters = pd.Series(cluster_id_list, index=features_df.index)
    return clusters


def _series_to_bit_vect(arr: pd.Series) -> IntSparseIntVect:
    """Convert a pd.Series to an RDKit IntSparseIntVect."""
    fp_size = len(arr)
    count_vect = IntSparseIntVect(fp_size)
    for index, count in enumerate(arr.to_list()):
        int_count = int(count)
        if int_count > 0:
            count_vect[index] = int_count

    return count_vect
