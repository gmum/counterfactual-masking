import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.DataStructs as DataStructs
from torch_geometric.loader import DataLoader as GraphDataLoader


class PseudoModel:
    def __init__(self, model, featurizer, device):
        self.model = model
        self.featurizer = featurizer
        self.device = device

    def forward(self, smiles):
        data = self.featurizer.process_single(mol=Chem.MolFromSmiles(smiles), y=0, smiles=smiles)
        data = data.to(self.device)
        if len(data.edge_index) == 0:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
        with torch.no_grad():
            output = self.model.predict(self.model, data)

        return output.cpu().numpy()

    def __call__(self, smiles):
        return self.forward(smiles)

def similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def validity(pred1, pred2):
    if pred1 < 0.5 and pred2 < 0.5:
        return False
    elif pred1 >= 0.5 and pred2 >= 0.5:
        return False
    else:
        return True

def diversity(collection):
    pairwise_sim = 0
    count = 0
    for i in range(len(collection)):
        for j in range(i + 1, len(collection)):
            pairwise_sim += similarity(collection[i], collection[j])
            count += 1
    avg_similarity = pairwise_sim / count
    return 1 - avg_similarity

def fast_diversity(collection):
    fingerprints = smiles_to_fingerprints(collection)
    n = len(fingerprints)
    sim_sum = 0.0
    count = 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
        dissim = [1 - s for s in sims]
        sim_sum += sum(dissim)
        count += len(dissim) 
    return sim_sum / count if count > 0 else 0.0

def smiles_to_fingerprints(smiles_list, radius=2, nBits=2048):
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprints.append(fp)
    return fingerprints
    

def counterfactuals_metrics(org_smiles, org_pred, counterfactuals_list, counterfactuals_predictions):
    samples_validity = 0
    samples_similarity = 0
    samples_diversity = 0

    for sample_smiles, sample_pred in zip(counterfactuals_list, counterfactuals_predictions):
        if validity(sample_pred, org_pred):
            samples_validity += 1
        samples_similarity += similarity(sample_smiles, org_smiles)

    num_samples = len(counterfactuals_list)
    if num_samples > 0:
        samples_diversity = fast_diversity(counterfactuals_list)
        samples_validity /= num_samples
        samples_similarity /= num_samples

    return samples_diversity, samples_validity, samples_similarity


def smiles_predictions(model, featurizer, smiles_list, batch_size=64, device="cpu"):
    mol_data = []
    valid_indices = []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) > 1:
            continue
        data = featurizer.process_single(mol, 1)
        mol_data.append(data)
        valid_indices.append(idx)

    predictions = [None] * len(smiles_list)

    if not mol_data:
        return predictions

    mol_loader = GraphDataLoader(mol_data, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()
    with torch.no_grad():
        batch_preds = []
        for data in mol_loader:
            data = data.to(device)
            preds = model.predict(model, data).flatten()
            batch_preds.extend(preds.cpu().numpy())

    for i, idx in enumerate(valid_indices):
        predictions[idx] = batch_preds[i]

    return predictions

def get_ring_atom_indices(smiles, atom_index):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    if min(atom_index) < 0 or max(atom_index) >= mol.GetNumAtoms():
        raise IndexError("Atom index out of range.")

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()  
    
    atoms_in_same_ring = set()
    for ring in atom_rings:
        if any(idx in ring for idx in atom_index):
            atoms_in_same_ring.update(ring)

    return sorted(atoms_in_same_ring)
    
    
def delete_bonds(smiles, top_20_percent_edges, bonds_dict):
    mol = Chem.MolFromSmiles(smiles)
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)  
    except Chem.rdchem.KekulizeException as e:
        print(f"Sanitization failed: {e}")
        raise
    editable_mol = Chem.EditableMol(mol)
    for idx in range(len(top_20_percent_edges)):
        atom1 = bonds_dict[0][top_20_percent_edges[idx]].item()
        atom2 = bonds_dict[1][top_20_percent_edges[idx]].item()
        editable_mol.RemoveBond(atom1, atom2)
    new_mol = editable_mol.GetMol()

    try:
        Chem.SanitizeMol(new_mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)  
        Chem.Kekulize(new_mol, clearAromaticFlags=True)  
    except Chem.rdchem.KekulizeException as e:
        print(f"Sanitization failed after atom removal: {e}")
        raise
    smiles = Chem.MolToSmiles(new_mol)
    return [smiles]
    
def find_closest_counterfactual(example_smiles, candidates_list, model, number_of_counterfactuals=3):
    original_pred = model(example_smiles)
    counterfactuals = []

    for candidate in candidates_list:
        if candidate.smiles == example_smiles:
            continue

        pred = model(candidate.smiles)
        # Check if prediction flips class
        if (original_pred >= 0.5 and pred < 0.5) or (original_pred < 0.5 and pred >= 0.5):
            sim = similarity(example_smiles, candidate.smiles)
            counterfactuals.append((candidate.smiles, pred, sim))

    # Sort by similarity descending and return top N
    counterfactuals = sorted(counterfactuals, key=lambda x: x[2], reverse=True)
    return counterfactuals[:number_of_counterfactuals]