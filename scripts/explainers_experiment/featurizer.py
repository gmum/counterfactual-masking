from typing import List, Dict, Literal

import torch
from torch_geometric.data import Data
from rdkit import Chem

_VERSION: Literal['v1', 'v2'] = 'v1'

class Featurizer:

    PERMITTED_ATOM_TYPES = {
        'v1': 'C O N H S Cl F X'.split(),
        'v2': 'C N O P S F Cl Br I H'.split(),
    }
    PERMITTED_ATOM_HYBRIDIZATIONS = 'SP SP2 SP3 SP3D SP3D2'.split()
    PERMITTED_BOND_TYPES = list(map(str, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]))

    @staticmethod
    def one_of_k_encoding_unk_v1(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def one_hot_encoding_v1(quantity, arr_length=5):
        last_idx = arr_length - 1
        idx = quantity if quantity < last_idx else last_idx
        n_encoding = [0] * arr_length
        n_encoding[idx] = 1
        return n_encoding

    @staticmethod
    def one_hot_encoding_v2(value: str, permitted: List[str]) -> List[int]:
        """Converts a value to a one-hot vector."""
        return [
            int((value if value in permitted else '?') == item)
            for item in permitted + ['?']
        ]

    @staticmethod
    def num_atom_features() -> int:
        return 22 if _VERSION == 'v1' else 34

    @staticmethod
    def num_bond_features() -> int:
        return len(Featurizer.PERMITTED_BOND_TYPES + ['?'])

    @staticmethod
    def feature_counts_dict() -> Dict[str, int]:
        return {
            'n_node_features': Featurizer.num_atom_features(),
            'n_edge_features': Featurizer.num_bond_features(),
        }

    def __init__(self):
        pass

    def __call__(
        self,
        molecule: Chem.rdchem.Mol,
        y: float | int,
        smiles: str,
    ) -> Data:
        """
        Converts an RDKit molecule object to a PyTorch Geometric Data object.
        """
        # atom featurization
        node_attributes = [
            {
                'v1': torch.tensor(
                      Featurizer.one_of_k_encoding_unk_v1(atom.GetSymbol(), Featurizer.PERMITTED_ATOM_TYPES['v1'])
                    + Featurizer.one_hot_encoding_v1(atom.GetTotalNumHs())
                    + Featurizer.one_hot_encoding_v1(len(atom.GetNeighbors()))
                    + [int(atom.IsInRing())]
                    + [int(atom.IsInRingSize(5))]
                    + [int(atom.IsInRingSize(6))]
                    + [int(atom.GetIsAromatic())],
                    dtype=torch.float
                ),
                'v2': torch.tensor(
                      Featurizer.one_hot_encoding_v2(atom.GetSymbol(), Featurizer.PERMITTED_ATOM_TYPES['v2'])
                    + Featurizer.one_hot_encoding_v2(atom.GetTotalNumHs(), list(range(5)))
                    + Featurizer.one_hot_encoding_v2(len(atom.GetNeighbors()), list(range(1, 6)))
                    + [int(atom.IsInRing())]
                    + [int(atom.IsInRingSize(5))]
                    + [int(atom.IsInRingSize(6))]
                    + [int(atom.GetIsAromatic())]
                    + Featurizer.one_hot_encoding_v2(str(atom.GetHybridization()), Featurizer.PERMITTED_ATOM_HYBRIDIZATIONS)
                    + [atom.GetFormalCharge()],
                    dtype=torch.float
                )
            }[_VERSION]
            for atom in molecule.GetAtoms()
        ]
        assert len(node_attributes[0]) == Featurizer.num_atom_features(), \
            f"{len(node_attributes[0])} != {Featurizer.num_atom_features()}"

        # bond featurization & connectivity
        edge_indices, edge_attributes = [], []
        for bond in molecule.GetBonds():
            begin_i, end_i = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.append((begin_i, end_i))
            edge_indices.append((end_i, begin_i))

            bond_type_one_hot = Featurizer.one_hot_encoding_v2(str(bond.GetBondType()), Featurizer.PERMITTED_BOND_TYPES)
            edge_attributes.append(bond_type_one_hot)
            edge_attributes.append(bond_type_one_hot)  # same features for the reverse edge
        assert len(edge_attributes[0]) == Featurizer.num_bond_features()

        return Data(
            x=torch.stack(node_attributes),
            y=torch.tensor([y], dtype=torch.float),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attributes, dtype=torch.float),
            smiles=Chem.CanonSmiles(smiles)
        )
