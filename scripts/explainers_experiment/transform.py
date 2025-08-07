from typing import Literal, List, Dict

import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem

from featurizer import Featurizer


class Transform:

    def __init__(self, featurizer: Featurizer, min_heavy_atoms: int, protonate: bool):
        self.ION_SMILES = set(
            ' [H+] [OH-] '
            ' [Cl-] [Na+] [K+] [Li+] [Br-] [I-] [F-] '
            ' [Ca+2] [Mg+2] [Zn+2] [Fe+2] [Fe+3]  [Al+2] [Al+3] [Cd+2] [Cr+3] [Hg+2] [Lu+3] [Pd+2] [Sn+2] [Mo+2] '
            ' [Zn] [Sr] [Co] [Cr] [Cu] [Mn] '
            ''.split()
        )
        self.featurizer = featurizer
        self.min_heavy_atoms = min_heavy_atoms
        self.protonate = protonate


    def _cleaned(self, molecule: Chem.rdchem.Mol) -> Chem.rdchem.Mol | None:
        '''Remove ions/salts and returns the largest molecule if not too small.'''

        filtered_fragments = [
            fragment
            for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
            if Chem.MolToSmiles(fragment) not in self.ION_SMILES
        ]
        if not filtered_fragments:
            return None

        largest_molecule = max(filtered_fragments, key=lambda m: m.GetNumHeavyAtoms())
        if largest_molecule.GetNumHeavyAtoms() < self.min_heavy_atoms:
            return None

        largest_molecule.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(largest_molecule)

        if self.protonate:
            return Chem.AddHs(largest_molecule)

        return largest_molecule


    def _to_geometric_dataset(
        self,
        dataframe: pd.DataFrame,
        smiles_col_name: str = 'Drug',
        target_col_name: str = 'Y',
    ) -> List[Data]:
        graphs = []
        for smiles, target in dataframe[
            [smiles_col_name, target_col_name]
        ].itertuples(index=False):
            molecule = Chem.MolFromSmiles(smiles, sanitize=False)
            if (clean_molecule := self._cleaned(molecule)) is not None:
                clean_molecule_smiles = Chem.CanonSmiles(Chem.MolToSmiles(clean_molecule))
                graph = self.featurizer(clean_molecule, y=target, smiles=clean_molecule_smiles)
                graphs.append(graph)
        return graphs


    def __call__(
        self,
        split_dataset: Dict[Literal['train', 'valid', 'test'], pd.DataFrame],
    ) -> Dict[Literal['train', 'valid', 'test'], Data]:
        return {
            partition_name: self._to_geometric_dataset(dataframe)
            for partition_name, dataframe in split_dataset.items()
        }
