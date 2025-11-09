from rdkit import Chem
from rdkit.Chem import SaltRemover, Crippen
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
import numpy as np
import json
from collections import defaultdict
from source.linksGenerator import find_anchors

remover = SaltRemover.SaltRemover()

def remove_salts_and_keep_largest(mol):
    """
    Removes salts from a molecule and keeps the largest fragment.
    """
    fragments = Chem.GetMolFrags(remover.StripMol(mol), asMols=True)
    if not fragments:
        return None
    largest_mol = max(fragments, key=lambda m: m.GetNumAtoms())
    return largest_mol

def process_dataset(dataset,featurizer):
    """
    Processes a dataset by removing salts from each molecule and re-featurizing.
    """
    new_dataset = []
    for i in range(len(dataset)):
        mol = Chem.MolFromSmiles(dataset[i].smiles)
        largest_mol = remove_salts_and_keep_largest(mol)
        if largest_mol is not None:
            smiles = Chem.MolToSmiles(largest_mol)
            new_dataset.append(featurizer.process_single(largest_mol, dataset[i].y , smiles))
    return new_dataset

def check_connection_between_atoms(common, atom_list):
    """
    Filters molecules by ensuring they maintain the bonding structure of a common substructure.
    """
    correct = []
    highlight_mol = Chem.MolFromSmiles(common)  
    bondLists = []
    for bond in highlight_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bondLists.append((begin_idx,end_idx))
    for smile in atom_list:  
        mol = Chem.MolFromSmiles(smile)  
        if mol:
            highlight_atoms = mol.GetSubstructMatch(highlight_mol) 
            highlight_atoms_changed = list(highlight_atoms)
            fail = False
            for atom in mol.GetAtoms():
                if atom.GetIdx() in highlight_atoms:
                    highlight_atoms_changed.remove(atom.GetIdx())
                    for bond in atom.GetBonds():
                        if bond.GetBeginAtomIdx() in highlight_atoms_changed or bond.GetEndAtomIdx() in highlight_atoms_changed:
                            begin_index = highlight_atoms.index(bond.GetBeginAtomIdx())
                            end_index = highlight_atoms.index(bond.GetEndAtomIdx())
                            if (begin_index,end_index) not in bondLists and (end_index,begin_index) not in bondLists:
                                fail = True
                                break
                    highlight_atoms_changed.append(atom.GetIdx())
            if not fail:
                correct.append(smile)
    return correct


def is_fully_connected(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol and len(Chem.GetMolFrags(mol)) == 1

def generate_universal_smile(smile):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile), canonical=True)


def choose_highest_prediction_pair(smiles_list, model, featurizer, device, batch_size=64):
    """
    Finds the pair of molecules with the highest difference in predictions.
    """
    mols = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            mols.append((mol, sm))
    # Using dummy label '0' during inference
    data_list = [featurizer.process_single(mol, 0, sm) for mol, sm in mols]

    preds = []
    model.eval()
    with torch.no_grad():
        loader = GraphDataLoader(data_list, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            batch_preds = model.predict(model, batch).flatten().cpu().numpy()
            preds.extend(batch_preds)

    if len(preds) < 2:
        return None
    
    preds = np.array(preds)
    min_idx = np.argmin(preds)
    max_idx = np.argmax(preds)
    max_diff = preds[max_idx] - preds[min_idx]

    return (min_idx, max_idx, max_diff)

def process_pairs_dataset(path, model, featurizer, device, number_of_anchors=None, same_anchors=False):
    """
    Loads a dataset containing pairs of molecules, processes them, and identifies the pair with the highest prediction difference.
    Parameters:
    - number_of_anchors: Specifies the anchor filtering criteria. 
        * If None, all anchors are considered.
        * If 1, only molecules with exactly one anchor to the masked part are considered.
        * If >1, molecules must have at least the specified number of anchors.
    - same_anchors: If True, only pairs with identical anchors in the masked part are considered.
    """
    print("Loading pairs dataset...")
    with open(path, 'r') as f:
        data = json.load(f)
        # data = [item for item in data if len(item[1]) > 1]
    data = [[key, value] for key, value in data.items()] # Running on superstructures_dataset
    print(f"Data length: {len(data)}")
    print("Choosing pairs...")
    canonical_data = []
    for item in data:
        canonical_smiles = generate_universal_smile(item[0])
        canonical_data.append([canonical_smiles, item[1]])

    merged_data = defaultdict(list)

    for smiles, molecules in canonical_data:
        merged_data[smiles].extend(molecules)

    merged_canonical_data = [[smiles, molecules] for smiles, molecules in merged_data.items()]

    for i in range(len(merged_canonical_data)):
        merged_canonical_data[i][1] = check_connection_between_atoms(merged_canonical_data[i][0], merged_canonical_data[i][1])

    new_set = {}
    for i, (main_smile, mol_list) in enumerate(merged_canonical_data):
        filtered = [generate_universal_smile(m) for m in mol_list if is_fully_connected(m) and Chem.MolFromSmiles(m).HasSubstructMatch(Chem.MolFromSmiles(main_smile)) and generate_universal_smile(m) != main_smile]
        new_filtered = []
        for m in filtered:
            mol = Chem.MolFromSmiles(m)
            common_atoms = mol.GetSubstructMatch(Chem.MolFromSmiles(main_smile))
            all_atoms = list(range(mol.GetNumAtoms()))
            non_common_atoms = [atom for atom in all_atoms if atom not in common_atoms]
            anchors = find_anchors(mol, non_common_atoms, original_indices=True)
            if number_of_anchors is None:
                if len(anchors) > 0:
                    new_filtered.append(m)
            elif number_of_anchors == 1:
                if len(anchors) == number_of_anchors:
                    new_filtered.append(m)
            elif number_of_anchors > 1:
                if len(anchors) >= number_of_anchors:
                    new_filtered.append(m)
        if len(new_filtered) < 1:
            continue
        if same_anchors == True:
            dictionary_of_molecules = {}
            for idx, smile in enumerate(new_filtered):
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    common_atoms = mol.GetSubstructMatch(Chem.MolFromSmiles(main_smile))
                    all_atoms = list(range(mol.GetNumAtoms()))
                    non_common_atoms = [atom for atom in all_atoms if atom not in common_atoms]
                    anchors = find_anchors(mol, non_common_atoms, original_indices=True, diffLinker=False)
                    index_of_anchors_in_common_part = set()
                    for anchor in anchors:
                        index_of_anchors_in_common_part.add(common_atoms.index(anchor))
                    org_anchors_tuple = tuple(index_of_anchors_in_common_part) 
                    if org_anchors_tuple not in dictionary_of_molecules:
                        dictionary_of_molecules[org_anchors_tuple] = [idx]
                    else:
                        dictionary_of_molecules[org_anchors_tuple].append(idx)
            #Choose set of anchors with the maximum number of molecules
            max_index = max(dictionary_of_molecules, key=lambda k: len(dictionary_of_molecules[k]))
            filtered_mols = [new_filtered[idx] for idx in dictionary_of_molecules[max_index]]
            new_filtered = filtered_mols

        pair = choose_highest_prediction_pair(new_filtered, model, featurizer, device)
        if pair and pair[2] > 1e-6:
            pair_data = [new_filtered[pair[0]], new_filtered[pair[1]]]
            if main_smile not in new_set or pair[2] > new_set[main_smile][1]:
                new_set[main_smile] = [pair_data, pair[2]]
    pairs_array = [[key, value[0], value[1]] for key, value in new_set.items()]
    print(f"Pairs array length: {len(pairs_array)}")
    return pairs_array