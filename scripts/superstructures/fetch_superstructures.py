import os
import argparse
import json
from tqdm import tqdm
from rdkit import Chem
from tdc.single_pred import ADME
from collections import defaultdict

from source.featurizers.graphs import Featurizer2D
from source.dataset_cleaner import process_dataset, check_connection_between_atoms, generate_universal_smile, is_fully_connected
from source.superstructures_creator import get_substructures, remove_attachment_points, fetch_superstructure_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="./data_pubchem",  help="Path to the output folder.")
    parser.add_argument("--dataset_name", type=str, default="common_substructure_pair_dataset", help="Name of the output dataset file.")
    args = parser.parse_args()

    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataset = "Solubility_AqSolDB"
    data = ADME(dataset)
    split = data.get_split(method="random", seed=42)

    featurizer = Featurizer2D('Y', 'Drug')
    split_data = featurizer(
        split=split,
        path=f'./data/splits/{dataset}',
    )

    x_test = split_data["test"]
    # Remove salts from each molecule
    x_test = process_dataset(x_test, featurizer)

    dict_of_similar = {}

    print("Fetching superstructure data...")
    for i in tqdm(range(len(x_test)), desc="Processing molecules"):
        substructures = get_substructures(x_test[i].smiles)
        for sup in substructures:
            clean_smiles = remove_attachment_points(sup)
            mol = Chem.MolFromSmiles(clean_smiles)
            if mol is None:
                continue
            num_atoms = mol.GetNumAtoms()
            # Get superstructrures only for molecules with more than 5 atoms
            if num_atoms < 5:
                continue
            try:
                smiles_list = fetch_superstructure_data(clean_smiles)
            except Exception as e:
                print(f"Error fetching substructure data: {e}")
                continue
            if len(smiles_list) == 0:
                continue
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            if canonical_smiles not in dict_of_similar:
                dict_of_similar[canonical_smiles] = smiles_list
            else:
                dict_of_similar[canonical_smiles].extend(smiles_list)

    print("Final filtering...")
    data = [[key, value] for key, value in dict_of_similar.items()]

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
        if main_smile not in new_set:
            new_set[main_smile] = filtered
        else:
            new_set[main_smile].extend(filtered)

    with open(os.path.join(output_folder, f'{args.dataset_name}.json'), 'w') as f:
        json.dump(new_set, f)

if __name__ == "__main__":
    main()