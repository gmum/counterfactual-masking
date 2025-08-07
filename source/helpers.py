from rdkit import Chem

def get_extended_ring_atom_indices(smiles, atom_indices):
    """
    Returns an extended list of atom indices, including all atoms that are in the same ring as any of the specified atom indices.
    """
    if atom_indices == []:
        return []
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    if min(atom_indices) < 0 or max(atom_indices) >= mol.GetNumAtoms():
        raise IndexError("Atom index out of range.")

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()  
    
    atoms_in_same_ring = set(atom_indices) 
    for ring in atom_rings:
        if any(idx in ring for idx in atom_indices):
            atoms_in_same_ring.update(ring)

    prev_atoms = set()
    while atoms_in_same_ring != prev_atoms:
        prev_atoms = set(atoms_in_same_ring)
        for ring in atom_rings:
            if any(idx in prev_atoms for idx in ring):
                atoms_in_same_ring.update(ring)

    return sorted(atoms_in_same_ring)