import torch

def feature_zeroing(data, masked_atoms):
    """
    Masking features in the input data.
    """
    output = data.clone()
    for i in masked_atoms:
        output.x[i] = torch.zeros_like(output.x[i])
    return output

#TODO: Implement the edge_changer function
def edge_changer(data):
    """
    Changing edges in the input data.
    """
    output = data.clone()

    print("Not implemented yet")
    return output

def atoms_remover(data, atom_indices):
    """
    Deleting atoms in the input data.
    """
    output = data.clone()
    atom_indices.sort()

    for i in range(len(output.edge_index[0])):
        # Selecting edges that contain the atom to be deleted
        if output.edge_index[0][i] in atom_indices or output.edge_index[1][i] in atom_indices:
            output.edge_index[0][i] = -1
            output.edge_index[1][i] = -1
    for i in range(len(output.edge_index[0])):
        # Updating the indices of the atoms
        if output.edge_index[0][i] > atom_indices[0]:
            decrease = 0
            for j in range(len(atom_indices)):
                if output.edge_index[0][i] > atom_indices[j]:
                    decrease += 1
                else: 
                    break
            output.edge_index[0][i] -= decrease
        if output.edge_index[1][i] > atom_indices[0]:
            decrease = 0
            for j in range(len(atom_indices)):
                if output.edge_index[1][i] > atom_indices[j]:
                    decrease += 1
                else: 
                    break
            output.edge_index[1][i] -= decrease
    # Deleting the edges that contain the atom to be deleted
    mask = (output.edge_index >= 0).all(dim=0)
    output.edge_index = output.edge_index[:, mask]


    mask = torch.ones(output.x .size(0), dtype=torch.bool)
    mask[torch.tensor(atom_indices)] = False
    # Apply the mask to filter rows
    output.x  = output.x[mask]

    return output