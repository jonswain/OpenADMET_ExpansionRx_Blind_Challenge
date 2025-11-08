"""Enhanced feature engineering for chemical property prediction."""

import logging

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Descriptors3D import NPR1, NPR2, PMI1, PMI2, PMI3
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.DataStructs import IntSparseIntVect
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from .config import (
    MORGAN_FINGERPRINT_RADIUS,
    MORGAN_FINGERPRINT_SIZE,
    RDKIT_FINGERPRINT_SIZE,
)

log = logging.getLogger(__name__)


class EnhancedFeatureGenerator:
    """Generate comprehensive molecular features for chemical property prediction."""

    def __init__(self):
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_FINGERPRINT_RADIUS, fpSize=MORGAN_FINGERPRINT_SIZE
        )
        self.rdk_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=RDKIT_FINGERPRINT_SIZE
        )
        self.topological_torsion_gen = (
            rdFingerprintGenerator.GetTopologicalTorsionGenerator()
        )
        self.atom_pair_gen = rdFingerprintGenerator.GetAtomPairGenerator()
        self.pharm_factory = Gobbi_Pharm2D.factory

    def generate_all_features(self, smiles_list: list[str]) -> pd.DataFrame:
        """Generate comprehensive molecular features.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with all molecular features
        """
        log.info("Generating comprehensive molecular features")

        mols = [
            Chem.MolFromSmiles(s)
            for s in tqdm(smiles_list, desc="Converting SMILES to Mols")
        ]

        features_dict = {}
        features_dict.update(self._generate_morgan_fingerprints(mols))
        features_dict.update(self._generate_other_fingerprints(mols))
        features_dict.update(self._generate_2d_descriptors(mols))
        features_dict.update(self._generate_3d_descriptors(mols))
        features_dict.update(self._generate_pharmacophore_features(mols))
        features_dict.update(self._generate_estate_fingerprints(mols))
        combined_df = pd.DataFrame(features_dict, index=smiles_list)
        if combined_df.isnull().values.any():
            log.warning("Some features contain NaN values")
            null_mask = combined_df.isnull().sum() > 0
            log.warning(combined_df.columns[null_mask].tolist())
            combined_df = combined_df.fillna(0)

        log.info(
            f"Generated {combined_df.shape[1]} features for {len(smiles_list)} molecules"
        )
        return combined_df

    def _generate_morgan_fingerprints(self, mols: list[Chem.Mol]) -> dict[str, list]:
        """Generate Morgan fingerprints with different radii."""
        features = {}
        fps = [
            self.morgan_gen.GetCountFingerprintAsNumPy(mol)
            for mol in tqdm(mols, desc="Calculating Morgan FPs")
        ]
        for i in range(MORGAN_FINGERPRINT_SIZE):
            features[f"fp_{i}"] = [fp[i] for fp in fps]
        return features

    def _generate_other_fingerprints(self, mols: list[Chem.Mol]) -> dict[str, list]:
        """Generate various other fingerprint types."""
        features = {}

        # MACCS keys
        maccs_fps = [
            MACCSkeys.GenMACCSKeys(mol)
            for mol in tqdm(mols, desc="Calculating MACCS keys")
        ]
        for i in range(167):
            features[f"maccs_{i}"] = [fp[i] for fp in maccs_fps]

        # RDKit fingerprints
        rdk_fps = [
            self.rdk_gen.GetCountFingerprintAsNumPy(mol)
            for mol in tqdm(mols, desc="Calculating RDKit FPs")
        ]
        for i in range(RDKIT_FINGERPRINT_SIZE):
            features[f"rdkit_fp_{i}"] = [fp[i] for fp in rdk_fps]

        # Atom pair fingerprints
        ap_fps = [
            self.atom_pair_gen.GetCountFingerprintAsNumPy(mol)
            for mol in tqdm(mols, desc="Calculating Atom Pair FPs")
        ]
        for i in range(2048):
            features[f"atom_pair_{i}"] = [fp[i] for fp in ap_fps]

        # Topological torsion fingerprints
        tt_fps = [
            self.topological_torsion_gen.GetCountFingerprintAsNumPy(mol)
            for mol in tqdm(mols, desc="Calculating Topological Torsion FPs")
        ]
        for i in range(2048):
            features[f"topo_torsion_{i}"] = [fp[i] for fp in tt_fps]

        return features

    def _generate_2d_descriptors(self, mols: list[Chem.Mol]) -> dict[str, list]:
        """Generate 2D molecular descriptors."""
        descriptors: list = []

        for mol in tqdm(mols, desc="Calculating 2D descriptors"):
            desc = {}
            for name, func in Descriptors._descList:
                if name == "Ipc":  # Skip problematic descriptor
                    continue
                try:
                    desc[name] = func(mol)
                except Exception:
                    desc[name] = np.nan

            desc.update(
                {
                    "num_aliphatic_carbocycles": rdMolDescriptors.CalcNumAliphaticCarbocycles(
                        mol
                    ),
                    "num_aliphatic_heterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles(
                        mol
                    ),
                    "num_aromatic_carbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles(
                        mol
                    ),
                    "num_aromatic_heterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(
                        mol
                    ),
                    "num_saturated_carbocycles": rdMolDescriptors.CalcNumSaturatedCarbocycles(
                        mol
                    ),
                    "num_saturated_heterocycles": rdMolDescriptors.CalcNumSaturatedHeterocycles(
                        mol
                    ),
                    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
                    "num_bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(
                        mol
                    ),
                    "num_spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
                }
            )

            descriptors.append(desc)

        all_keys = set()
        for d in descriptors:
            all_keys.update(d.keys())

        features = {}
        for key in all_keys:
            features[key] = [d.get(key, np.nan) for d in descriptors]

        return features

    def _generate_3d_descriptors(self, mols: list[Chem.Mol]) -> dict[str, list]:
        """Generate 3D molecular descriptors (requires conformer generation)."""
        features = {}
        descriptors_3d = []

        for mol in tqdm(mols, desc="Calculating 3D descriptors"):
            desc_3d: dict = {}
            try:
                # Add hydrogens and generate conformer
                mol_h = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_h, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol_h)

                # Calculate 3D descriptors
                desc_3d.update(
                    {
                        "NPR1": NPR1(mol_h),
                        "NPR2": NPR2(mol_h),
                        "PMI1": PMI1(mol_h),
                        "PMI2": PMI2(mol_h),
                        "PMI3": PMI3(mol_h),
                        "spherocity": rdMolDescriptors.CalcSpherocityIndex(mol_h),
                        "asphericity": rdMolDescriptors.CalcAsphericity(mol_h),
                        "eccentricity": rdMolDescriptors.CalcEccentricity(mol_h),
                        "inertial_shape_factor": rdMolDescriptors.CalcInertialShapeFactor(
                            mol_h
                        ),
                    }
                )

            except Exception as e:
                # If 3D generation fails, set to NaN
                log.warning("3D descriptor calculation failed for a molecule: %s", e)
                desc_3d = {
                    "NPR1": np.nan,
                    "NPR2": np.nan,
                    "PMI1": np.nan,
                    "PMI2": np.nan,
                    "PMI3": np.nan,
                    "spherocity": np.nan,
                    "asphericity": np.nan,
                    "eccentricity": np.nan,
                    "inertial_shape_factor": np.nan,
                }

            descriptors_3d.append(desc_3d)

        # Convert to features dictionary
        all_keys: set = set()
        for d in descriptors_3d:
            all_keys.update(d.keys())

        for key in all_keys:
            features[key] = [d.get(key, np.nan) for d in descriptors_3d]

        return features

    def _generate_pharmacophore_features(self, mols: list[Chem.Mol]) -> dict[str, list]:
        """Generate pharmacophore fingerprints."""
        features = {}

        pharm_fps = []
        for mol in tqdm(mols, desc="Calculating pharmacophore fingerprints"):
            try:
                fp = Generate.Gen2DFingerprint(mol, self.pharm_factory)
                folded_fp = DataStructs.FoldFingerprint(fp, int(39972 / 1024))
                # Convert to numpy array
                arr = np.zeros(1024)
                for i in range(1024):
                    if folded_fp.GetBit(i):
                        arr[i] = 1
                pharm_fps.append(arr)
            except Exception:
                pharm_fps.append(np.zeros(1024))

        for i in range(1024):
            features[f"pharm_{i}"] = [fp[i] for fp in pharm_fps]

        return features

    def _generate_estate_fingerprints(self, mols: list[Chem.Mol]) -> dict[str, list]:
        """Generate E-State fingerprints."""
        features = {}

        estate_fps = []
        for mol in tqdm(mols, desc="Calculating E-State fingerprints"):
            try:
                fp = Fingerprinter.FingerprintMol(mol)
                arr = np.zeros(79)
                for i in range(79):
                    arr[i] = fp.GetVal(i)
                estate_fps.append(arr)
            except Exception:
                estate_fps.append(np.zeros(79))

        for i in range(79):
            features[f"estate_{i}"] = [fp[i] for fp in estate_fps]

        return features


def generate_enhanced_features(
    smiles_list: list[str],
) -> pd.DataFrame:
    """Generate enhanced molecular features.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        DataFrame with enhanced molecular features
    """
    generator = EnhancedFeatureGenerator()
    return generator.generate_all_features(smiles_list)


def calculate_butina_clusters(
    features_df: pd.DataFrame,
    cutoff: float = 0.35,
    fpSize: int = MORGAN_FINGERPRINT_SIZE,
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
    for i in tqdm(range(1, nfps), desc="Calculating distances for clustering"):
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
