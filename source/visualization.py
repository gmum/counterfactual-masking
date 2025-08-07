from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
import numpy as np
from IPython.display import Image
import py3Dmol

def visualize_contrastive_saliency(mol, saliency_map, threshold=None):
    if threshold is None:
        saliency_map = np.array(saliency_map)
        norm_saliency = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        # Assign colors to atoms based on saliency scores
        atom_colors = {i: (1.0, 1.0 - norm_saliency[i], 1.0 - norm_saliency[i]) for i in range(len(norm_saliency))}

        d = rdMolDraw2D.MolDraw2DCairo(300, 300)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_colors.keys(), highlightAtomColors=atom_colors)

        d.FinishDrawing()
        png = d.GetDrawingText()
    else:
        atom_colors = {}
        for i in range(mol.GetNumAtoms()):
            if saliency_map[i] > threshold:
                atom_colors[i] = (1.0, 0.0, 0.0)

        d = rdMolDraw2D.MolDraw2DCairo(300, 300)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_colors.keys(), highlightAtomColors=atom_colors)

        d.FinishDrawing()
        png = d.GetDrawingText()

    return Image(png)

def draw_molecule(mol, highlightAtoms=None, highlightAtomColors=None, highlightBonds=None, highlightBondColors=None):
    d = rdMolDraw2D.MolDraw2DCairo(300, 300)

    if highlightAtomColors is None and highlightAtoms is not None:
        highlightAtomColors = {i: (0.0, 0.5, 0.0) for i in highlightAtoms}

    if highlightBondColors is None and highlightBonds is not None:
        highlightBondColors = {i: (1.0, 1.0, 0.0) for i in highlightBonds}

    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlightAtoms, highlightAtomColors=highlightAtomColors, highlightBonds=highlightBonds, highlightBondColors=highlightBondColors)
    
    d.FinishDrawing()
    png = d.GetDrawingText()

    return Image(png)

def visualize_3D(mol, highlightAtoms=None):
    view = py3Dmol.view(width=400, height=400)
    view.addModel(Chem.MolToMolBlock(mol), "mol")
    view.setStyle({'stick': {}})
    if highlightAtoms is not None:
        for i in highlightAtoms:
            view.addStyle({'model': 0, 'serial': i}, {'sphere': {'color': 'green', 'radius': 0.5}})
    view.zoomTo()
    view.show()

def display_all_gen(folder, showNew=False, numberOfNew = 0):
    for i in range(5):
        gen_mol = Chem.MolFromMolFile(f"{folder}/gen_{i}.sdf")
        try:
            Chem.SanitizeMol(gen_mol)
            print(Chem.MolToSmiles(gen_mol))
            if showNew:
                num_atoms = gen_mol.GetNumAtoms()
                numberOfNew = min(numberOfNew, num_atoms)
                atoms = list(range(num_atoms))
                atoms_to_highlight = atoms[-int(numberOfNew):]
                visualize_3D(gen_mol,atoms_to_highlight) 
            else: 
                visualize_3D(gen_mol)  
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Molecule skipped: data_diff/{folder}/gen/gen_{i}.sdf")

def show_atom_number(mol, label="atomNote"):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol
