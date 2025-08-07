import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from source.featurizers.helpers import (IFeaturizer, one_hot_encoding,
                                     one_of_k_encoding_unk)

symbols = ["C", "O", "N", "H", "S", "Cl", "F", "X"]


class Featurizer2D(IFeaturizer):
    """
    Featurizer2D class

    This class is used to featurize the data for 2D models.
    """

    def __init__(self, y_column, smiles_col):
        super().__init__(y_column, smiles_col)

    def process(self, df):
        print("PROCESSING")
        graphs = []
        for _, row in df.iterrows():
            y = row[self.y_column]
            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            nodes, edge_index = self.transform_molecule(mol)
            graphs.append((nodes, y, edge_index, smiles))

        data = [
            Data(
                x=torch.FloatTensor(x),
                y=torch.FloatTensor([y]),
                edge_index=torch.LongTensor(edge_index),
                smiles=smiles,
            )
            for ((x, y, edge_index, smiles)) in graphs
        ]
        return data

    def transform_molecule(self, mol):
        edges = []
        for bond in mol.GetBonds():
            x_atoms = bond.GetBeginAtomIdx()
            y_atoms = bond.GetEndAtomIdx()
            edges.append((x_atoms, y_atoms))
            edges.append((y_atoms, x_atoms))
        edges = np.array(edges)

        nodes = []

        for atom in mol.GetAtoms():
            results = one_of_k_encoding_unk(atom.GetSymbol(), symbols)

            h_encoding = one_hot_encoding(atom.GetTotalNumHs())

            results = [*results, *h_encoding]

            results.append(atom.IsInRing())
            results.append(atom.GetIsAromatic())
            results.append(atom.IsInRingSize(5))
            results.append(atom.IsInRingSize(6))

            n_encoding = one_hot_encoding(len(atom.GetNeighbors()))
            results = [*results, *n_encoding]

            result = np.array(results, dtype=int)
            nodes.append(result)
        nodes = np.array(nodes)

        self.number_of_features = nodes.shape[1]

        return nodes, edges.T

    def get_nr_of_features(self):
        """
        Returns the number of features in the dataset. (size of input layer of the model)
        """
        return self.number_of_features

    def process_single(self, mol, y, smiles = None):
        nodes, edge_index = self.transform_molecule(mol)
        data = Data(
            x=torch.FloatTensor(nodes),
            y=torch.FloatTensor([y]),
            edge_index=torch.LongTensor(edge_index),
            smiles=smiles
        )
        return data
