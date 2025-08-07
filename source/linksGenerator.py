import os
import sys
import signal
import random
import subprocess
from math import prod
from itertools import product
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem
from reformat_data_obabel import load_rdkit_molecule
from crem.crem import mutate_mol

def find_anchors(mol, to_delete, bonds_dict = None, diffLinker = True, original_indices = False):
    """
    Identify anchor atoms (attachment points) of atoms to delete in a molecular structure.
    """
    highest_atom_index = mol.GetNumAtoms() - 1

    # Create a dictionary of bonds if not provided
    if bonds_dict is None:
        bonds_dict = create_graph_dictionary(mol)

    # Find the anchors
    if diffLinker:
        #Find unique neighbors (anchors) that are not in the to_delete list
        anchors_before_adaptation = set()
        for atom in to_delete:
            anchors_before_adaptation.update(bonds_dict.get(atom, []))
        anchors_before_adaptation.difference_update(to_delete)
    else:
        # Find all neighbors (anchors) that are not in the to_delete list (allows for duplicates)
        anchors_before_adaptation = []
        for atom_idx in to_delete:
            neighbors = bonds_dict.get(atom_idx, [])
            for neighbor in neighbors:
                if neighbor not in to_delete:
                    anchors_before_adaptation.append(neighbor)

    # Return the original indices of the anchors without adjusting for deleted atoms
    if original_indices:
        return [x+1 for x in anchors_before_adaptation] if diffLinker else anchors_before_adaptation
    
    # Adjust the indices after atom deletion
    sorted_to_delete = sorted(to_delete)
    index_adjustment = [0]*(highest_atom_index + 1)
    decrease = 0
    idx_in_delete = 0
    for idx in range(highest_atom_index + 1):
        if idx_in_delete < len(sorted_to_delete) and idx == sorted_to_delete[idx_in_delete]:
            decrease += 1
            idx_in_delete += 1
        index_adjustment[idx] = decrease
    anchors_index_after_deletion = list(anchors_before_adaptation)
    anchors_index_after_deletion.sort(reverse=False)
    for i in range(len(anchors_index_after_deletion)):
        anchors_index_after_deletion[i] -= index_adjustment[anchors_index_after_deletion[i]]

    return [x+1 for x in anchors_index_after_deletion] if diffLinker else anchors_index_after_deletion

def atomDeleter(mol, toDelete):
    """
    Delete specified atoms from a molecule.
    """
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)  
    except Chem.rdchem.KekulizeException as e:
        print(f"Sanitization failed: {e}")
        raise
    # Remove the specified atoms
    emol = Chem.EditableMol(mol)  
    for idx in sorted(toDelete, reverse=True):  
        emol.RemoveAtom(idx)
    new_mol = emol.GetMol()

    try:
        Chem.SanitizeMol(new_mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)  
        Chem.Kekulize(new_mol, clearAromaticFlags=True)  
    except Chem.rdchem.KekulizeException as e:
        print(f"Sanitization failed after atom removal: {e}")
        raise
    return new_mol

def dfs(node, graph, subset, visited):
    """
    Perform DFS to find all nodes in the same connected component.
    """
    stack = [node]
    component = []

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            component.append(current)
            stack.extend(neighbor for neighbor in graph.get(current, []) if neighbor in subset and neighbor not in visited)
    return component

def find_connections(subset, graph):
    """
    Find all connected components within a subset of nodes in a graph.
    """
    visited = set()
    connected_components = []

    for node in subset:
        if node not in visited:
            component = dfs(node, graph, subset, visited)
            connected_components.append(component)

    return connected_components

def create_graph_dictionary(mol):
    """
    Create a dictionary (adjacency list) representing the molecular graph.
    """
    bonds_dict = {}
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atom1 in bonds_dict:
            bonds_dict[atom1].append(atom2)
        else:
            bonds_dict[atom1] = [atom2]
        if atom2 in bonds_dict:
            bonds_dict[atom2].append(atom1)
        else:
            bonds_dict[atom2] = [atom1]
    return bonds_dict
    
def make_clear_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

def map_old_to_new_indices(mol, toDelete):
    """
    Create a mapping from old atom indices to new atom indices after deleting specified atoms.
    """
    highest_atom_index = mol.GetNumAtoms() - 1
    sorted_to_delete = sorted(toDelete)

    index_adjustment = [0] * (highest_atom_index + 1)
    decrease = 0
    idx_in_delete = 0

    for idx in range(highest_atom_index + 1):
        if idx_in_delete < len(sorted_to_delete) and idx == sorted_to_delete[idx_in_delete]:
            decrease += 1
            idx_in_delete += 1
        index_adjustment[idx] = decrease
        
    old_to_new = {idx: idx - index_adjustment[idx] for idx in range(highest_atom_index + 1) if idx not in toDelete}
    return old_to_new

def get_project_path(*args, base=None):
    if base:
        return os.path.join(base, *args)
    return os.path.join(*args)

def prepare_folders(folder, project_path):
    """
    Prepares the necessary subfolders by clearing them or creating if missing.
    """
    for subfolder in ["combined", "fragments", "gen", "output"]:
        make_clear_folder(get_project_path("data_diff", folder, subfolder, base=project_path))


def delete_Generate_diffLinker(mol, toDelete, folder):
    """
    Delete specified atoms from a molecule and generate missing parts using DiffLinker.
    """
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    prepare_folders(folder, project_path)

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    Chem.Kekulize(mol, clearAromaticFlags=True)
    # Create a bonds dictionary
    bonds_dict = create_graph_dictionary(mol)
    # Find connected components in the subset of atoms to delete
    connected_components = find_connections(toDelete, bonds_dict)
    # Map indices of atoms in the original molecule to the new indices after deletion
    old_to_new_index_map = map_old_to_new_indices(mol, toDelete)
    trueSmile = Chem.MolToSmiles(mol)

    # Generate missing fragments asynchronously
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, component in enumerate(connected_components):
            # Submit each component's processing as a separate task
            futures.append(executor.submit(process_component, idx, component, mol, folder, bonds_dict, project_path))
        # Wait for all tasks to complete before continuing
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in fragment generation: {e}")
                
    # Rebuild molecules using the generated fragments      
    for i in range(5):
        original_mol = atomDeleter(mol, toDelete)

        for idx, component in enumerate(connected_components):
            xyz_file_path = get_project_path("data_diff", folder, f"{idx}_diff_output", f"output_{i}_fragments_{idx}_.xyz", base=project_path)
            mol_from_xyz = Chem.MolFromXYZFile(xyz_file_path)

            if mol_from_xyz is None:
                print(f"Warning: Failed to load XYZ file {xyz_file_path}")
                continue

            # Keep only the generated new atoms
            num_atoms = mol_from_xyz.GetNumAtoms()
            atoms_to_keep = list(range(num_atoms - len(component), num_atoms))
            atoms_to_delete = [j for j in range(num_atoms) if j not in atoms_to_keep]
            emol = Chem.EditableMol(mol_from_xyz)
            for j in sorted(atoms_to_delete, reverse=True):
                emol.RemoveAtom(j)
                
            # Add the new atoms to the original molecule
            original_mol = Chem.CombineMols(original_mol, emol.GetMol())
            
        # Save the combined molecule
        combined_xyz_path = get_project_path("data_diff", folder, "combined", f"combined_molecule_{i}.xyz", base=project_path)
        Chem.MolToXYZFile(original_mol, combined_xyz_path)
        # Load the combined molecule and save it as SDF
        load_rdkit_molecule(combined_xyz_path, get_project_path("data_diff", folder, "gen", f"gen_{i}.sdf", base=project_path), trueSmile)
    
    # Fix bonds in generated molecules
    for i in range(5):
        try:
            gen_sdf_path = get_project_path("data_diff", folder, "gen", f"gen_{i}.sdf", base=project_path)
            generated_mol = Chem.MolFromMolFile(gen_sdf_path)

            if generated_mol is None:
                print(f"Warning: Failed to load generated molecule {i}")
                continue
            
            emol = Chem.EditableMol(generated_mol)
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                # Check if the bond atoms are in the toDelete list
                if atom1 in toDelete or atom2 in toDelete:
                    continue

                atom1_in_generated = old_to_new_index_map.get(atom1)
                atom2_in_generated = old_to_new_index_map.get(atom2)

                if atom1_in_generated is None or atom2_in_generated is None:
                    continue
                
                # Remove the bond if it exists in the generated molecule
                temp_mol = emol.GetMol()
                if temp_mol.GetBondBetweenAtoms(atom1_in_generated, atom2_in_generated) is not None:
                    emol.RemoveBond(atom1_in_generated, atom2_in_generated)
                # Restore the bond with the original bond type
                emol.AddBond(atom1_in_generated, atom2_in_generated, bond_type)

            if emol.GetMol().GetNumAtoms() != mol.GetNumAtoms():
                continue
            num_fragments = len(Chem.GetMolFrags(emol.GetMol()))
            if num_fragments > 1:
                continue
            try:
                Chem.SanitizeMol(emol.GetMol())
            except Exception as e:
                continue

            output_sdf_path = get_project_path("data_diff", folder, "output", f"gen_{i}.sdf", base=project_path)
            with Chem.SDWriter(output_sdf_path) as writer:
                writer.write(emol.GetMol())

        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue

    return None


def process_component(idx, component, mol, folder, bonds_dict, project_path):
    """
    Processes a single fragment:
    - Deletes atoms to isolate the fragment.
    - Finds anchor points.
    - Calls DiffLinker to generate the missing part.
    """
    try:
        new_mol = atomDeleter(mol, component)
        true_smile = Chem.MolToSmiles(new_mol)

        # Save the fragment as SDF
        sdf_path = get_project_path("data_diff", folder, "fragments", f"fragments_{idx}.sdf", base=project_path)
        diffLinker_path = get_project_path("DiffLinker", base=project_path)
        with Chem.SDWriter(sdf_path) as writer:
            writer.write(new_mol)
            
        # Find anchors
        anchors = find_anchors(mol, component, bonds_dict)
        anchors_str = ','.join(map(str, anchors))
        linker_size = len(component)

        command = [
            sys.executable, f"{diffLinker_path}/generate.py",
            "--fragments", sdf_path,
            "--model", f"{diffLinker_path}/models/zinc_difflinker_given_anchors.ckpt",
            "--anchors", anchors_str,
            "--linker_size", str(linker_size),
            "--output", get_project_path("data_diff", folder, f"{idx}_diff_output", base=project_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"DiffLinker generation failed for component {idx}:\n{result.stderr}")
    except Exception as e:
        print(f"Error processing component {idx}: {e}")


def crem_mutate_linker(mol, toDelete, radius = 1, min_max_inc = 3, max_replacements = None):
    """
    Delete specified atoms from a molecule and generate missing parts using Crem.
    """
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Merge components to mask if they share anchors
    components_to_delete = find_connections(toDelete, create_graph_dictionary(mol))
    merged_components_list = []
    anchors_org = []
    for component in components_to_delete:
        anchor = find_anchors(mol, component,bonds_dict=None, diffLinker=False, original_indices=True)
        if anchor not in anchors_org:
            anchors_org.append(anchor)
            merged_components_list.append(component)
        else:
            index = anchors_org.index(anchor)
            merged_components_list[index].extend(component)
    print("Number of components", len(merged_components_list))
    grown_mols = []
    for index, component in enumerate(merged_components_list):
        print(f"Component {index + 1}/{len(merged_components_list)}")
        mol_copy = mol
        anchor = find_anchors(mol, component,bonds_dict=None, diffLinker=False, original_indices=False)
        grow_mols = []
        try:
            Chem.SanitizeMol(mol_copy)
            print("Component", component)
            print(Chem.MolToSmiles(mol_copy))
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  
            try:
                grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=0, max_inc=0, max_replacements = max_replacements, symmetry_fixes=True))
            except TimeoutException as e:
                print(e)
            finally:
                signal.alarm(0) 
            if len(grow_mols) == 0:
                grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=0, max_inc=0, max_replacements = max_replacements, symmetry_fixes=False))
            
            number = 1
            # Gradually allow bigger replacements if still no results
            while len(grow_mols) == 0 and number < min_max_inc + 1:
                print("Change in size of the replacement", number)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  
                try:
                    grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=number, max_inc=number, max_replacements=max_replacements, symmetry_fixes = True))
                except TimeoutException as e:
                    print(e)
                finally:
                    signal.alarm(0)  
                if len(grow_mols) == 0:
                    grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=number, max_inc=number, max_replacements=max_replacements, symmetry_fixes = False))
                number += 1
            print("Number of grown mols", len(grow_mols))
            grown_mols.append(grow_mols)
        except Exception as e:
            print(f"Error during grow_mol: {e}")
            return []
        
    # Means that there is no grown mol
    if len(grown_mols) == 0:
        return []
    
    # Determine minimal number of mols generated across components
    how_many = float("inf")
    for grown_mol in grown_mols:
        how_many = min (len(grown_mol), how_many)
    print("How many", how_many)

    # Remove extra atoms from the original molecule
    mol_empty = atomDeleter(mol,toDelete) 

    output = []
    # Combine the original mol with new parts
    for index in range(how_many):
        atom_counts = []
        anchors_org_agg = []
        anchors_new_part_agg = []
        mols_to_combine = [mol_empty]
        atom_counts.append(mol_empty.GetNumAtoms())
        anchors_org = []
        anchors_new_part = []
        for idx_grow_mol,grow_mol in enumerate(grown_mols):
            mol_witout_part = atomDeleter(mol, merged_components_list[idx_grow_mol])

            mol_witout_part_indices = [atom.GetIdx() for atom in mol_witout_part.GetAtoms()]
            new = grow_mol[index]
            new_grow_mol = Chem.MolFromSmiles(new)

            Chem.SanitizeMol(mol_witout_part)
            Chem.SanitizeMol(new_grow_mol)
            substruct_match = new_grow_mol.GetSubstructMatch(mol_witout_part)

            grow_mol_indices = [atom.GetIdx() for atom in new_grow_mol.GetAtoms()]
            anchors_org = find_anchors(new_grow_mol, list(set(grow_mol_indices) - set(substruct_match)),bonds_dict=None, diffLinker = False, original_indices = True)
            anchors_new_part = list(find_anchors(new_grow_mol, list(set(substruct_match)),bonds_dict=None, diffLinker = False, original_indices = False))

            true_anchors_org = []
            for anchor in anchors_org:
                true_anchors_org.append(mol_witout_part_indices[substruct_match.index(anchor)])

            # Correct indices
            for i in range(len(true_anchors_org)):
                changer = 0
                for component in merged_components_list:
                    for value in component:
                        if value in merged_components_list[idx_grow_mol]:
                            continue
                        if value < true_anchors_org[i]:
                            changer += 1
                            
                true_anchors_org[i] -= changer

            new_part = atomDeleter(new_grow_mol, substruct_match)
            atom_counts.append(new_part.GetNumAtoms())
            mols_to_combine.append(new_part)

            anchors_org_agg.append(true_anchors_org)
            anchors_new_part_agg.append(anchors_new_part)

            # Combine molecules
            mol_objs = [Chem.EditableMol(mol_1).GetMol() for mol_1 in mols_to_combine]
            combined = reduce(Chem.CombineMols, mol_objs)
            editable_combined = Chem.EditableMol(combined)

        offsets = [0]
        for count in atom_counts[:-1]:
            offsets.append(offsets[-1] + count)
        editable_combined = Chem.EditableMol(combined)
        
        # Add bonds between the original molecule and the new parts
        for idx in range(len(anchors_org_agg)):
            for i in range(len(anchors_org_agg[idx])):
                editable_combined.AddBond(anchors_org_agg[idx][i], anchors_new_part_agg[idx][i] + offsets[idx+1], Chem.rdchem.BondType.SINGLE)
        try:
            combined_mol = editable_combined.GetMol()
            Chem.SanitizeMol(combined_mol)
        except Exception as e:
            print(f"Error during sanitization: {e}")
            continue
        if len(Chem.GetMolFrags(combined_mol)) > 1:
            print("Warning: Combined molecule has more than one fragment.")
        else:
            output.append(Chem.MolToSmiles(combined_mol))
    return output

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Mutate mol with symmetry fix time out.")


def crem_fragment_replacement(mol, toDelete, radius = 1, min_max_inc = 3, max_replacements = None, return_new_atom_indices = False):
    """
    Delete specified atoms from a molecule and generate missing parts using CReM.
    Parameters:
    ----------
    mol : rdkit.Chem.Mol
        The input molecule to be modified.
    toDelete : list of int
        A list of atom indices in the molecule that should be changed.
    radius : int, optional, default=1
        The radius of the chemical environment to consider for replacements.
    min_max_inc : int, optional, default=3
        The maximum change in the size of replacements allowed during mutation.
    max_replacements : int or None, optional, default=None
        The maximum number of replacements for a single connected part. If None, no limit is applied.
    return_new_atom_indices : bool, optional, default=False
        If True, returns the new atom indices of the generated molecules.
    Returns:
    -------
    list of str
        A list of SMILES strings representing the modified molecules generated 
        by combining the original molecule with the new parts.
    """
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Merge components that share the same anchors
    components_to_delete = find_connections(toDelete, create_graph_dictionary(mol))
    merged_components_list = []
    anchors_org = []
    for component in components_to_delete:
        anchor = find_anchors(mol, component, bonds_dict=None, diffLinker=False, original_indices=True)
        if anchor not in anchors_org:
            anchors_org.append(anchor)
            merged_components_list.append(component)
        else:
            index = anchors_org.index(anchor)
            merged_components_list[index].extend(component)
    print("Number of components:", len(merged_components_list))
    grown_mols = []
    for index, component in enumerate(merged_components_list):
        print(f"Component {index + 1}/{len(merged_components_list)}")
        mol_copy = mol
        anchor = find_anchors(mol, component,bonds_dict=None, diffLinker=False, original_indices=False)
        grow_mols = []
        try:
            Chem.SanitizeMol(mol_copy)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  
            try:
                grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=0, max_inc=0, max_replacements = max_replacements, symmetry_fixes=True))
            except TimeoutException as e:
                print(e)
            finally:
                signal.alarm(0) 
            if len(grow_mols) == 0:
                grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=0, max_inc=0, max_replacements = max_replacements, symmetry_fixes=False))
            
            number = 1
            # Gradually allow bigger replacements if still no results
            while len(grow_mols) == 0 and number < min_max_inc + 1:
                print("Change in size of the replacement", number)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  
                try:
                    grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=number, max_inc=number, max_replacements=max_replacements, symmetry_fixes = True))
                except TimeoutException as e:
                    print(e)
                finally:
                    signal.alarm(0)  
                if len(grow_mols) == 0:
                    grow_mols = list(mutate_mol(mol_copy, db_name= get_project_path("data", "chembl22_sa2.db", base=project_path), radius=radius, replace_ids = component,  min_inc=number, max_inc=number, max_replacements=max_replacements, symmetry_fixes = False))
                number += 1
            print("Number of grown mols", len(grow_mols))
            grown_mols.append(grow_mols)
        except Exception as e:
            print(f"Error during grow_mol: {e}")
            return []

    # Means that there is no growed mol
    if len(grown_mols) == 0:
        return []
    if len(grown_mols) == 1 and return_new_atom_indices == False:
        # If only one component, return the generated molecules directly
        return grown_mols[0] 
    
    mol_sizes = [len(mols) for mols in grown_mols]

    # Total number of possible combinations
    total_combinations = prod(mol_sizes)

    # Generate combinations
    if total_combinations <= 500:
        index_combinations = list(product(*(range(size) for size in mol_sizes)))
    else:
        sampled_indices = random.sample(range(total_combinations), 500)
        index_combinations = [flat_index_to_tuple(i, mol_sizes) for i in sampled_indices] 

    # Remove extra atoms from the original molecule
    mol_empty = atomDeleter(mol,toDelete) 
    output = []
    if return_new_atom_indices == True:
        new_atom_indices = []
    # Combine the original mol with new parts
    for index_tuple in index_combinations:
        atom_counts = []
        anchors_org_agg = []
        anchors_new_part_agg = []
        mols_to_combine = [mol_empty]
        atom_counts.append(mol_empty.GetNumAtoms())

        for idx_grow_mol, grow_index in enumerate(index_tuple):
            mol_witout_part = atomDeleter(mol, merged_components_list[idx_grow_mol])
            mol_witout_part_indices = [atom.GetIdx() for atom in mol_witout_part.GetAtoms()]
            new = grown_mols[idx_grow_mol][grow_index]
            new_grow_mol = Chem.MolFromSmiles(new)

            Chem.SanitizeMol(mol_witout_part)
            Chem.SanitizeMol(new_grow_mol)
            substruct_match = new_grow_mol.GetSubstructMatch(mol_witout_part)

            grow_mol_indices = [atom.GetIdx() for atom in new_grow_mol.GetAtoms()]
            anchors_org = find_anchors(new_grow_mol, list(set(grow_mol_indices) - set(substruct_match)),bonds_dict=None, diffLinker = False, original_indices = True)
            anchors_new_part = list(find_anchors(new_grow_mol, list(set(substruct_match)),bonds_dict=None, diffLinker = False, original_indices = False))

            true_anchors_org = []
            for anchor in anchors_org:
                true_anchors_org.append(mol_witout_part_indices[substruct_match.index(anchor)])

            # Correct indices
            for i in range(len(true_anchors_org)):
                changer = 0
                for component in merged_components_list:
                    for value in component:
                        if value in merged_components_list[idx_grow_mol]:
                            continue
                        if value < true_anchors_org[i]:
                            changer += 1
                            
                true_anchors_org[i] -= changer

            new_part = atomDeleter(new_grow_mol, substruct_match)
            atom_counts.append(new_part.GetNumAtoms())
            mols_to_combine.append(new_part)

            anchors_org_agg.append(true_anchors_org)
            anchors_new_part_agg.append(anchors_new_part)

            # Combine molecules
            mol_objs = [Chem.EditableMol(mol_1).GetMol() for mol_1 in mols_to_combine]
            combined = reduce(Chem.CombineMols, mol_objs)
            editable_combined = Chem.EditableMol(combined)

        offsets = [0]
        for count in atom_counts[:-1]:
            offsets.append(offsets[-1] + count)
        editable_combined = Chem.EditableMol(combined)
        # Add bonds between the original molecule and the new parts
        try:
            for idx in range(len(anchors_org_agg)):
                for i in range(len(anchors_org_agg[idx])):
                    editable_combined.AddBond(anchors_org_agg[idx][i], anchors_new_part_agg[idx][i] + offsets[idx+1], Chem.rdchem.BondType.SINGLE)
        except Exception as e:
            print(f"Error adding bonds: {e}")
            continue
        try:
            combined_mol = editable_combined.GetMol()
            Chem.SanitizeMol(combined_mol)
        except Exception as e:
            print(f"Error during sanitization: {e}")
            continue
        if len(Chem.GetMolFrags(combined_mol)) > 1:
            print("Warning: Combined molecule has more than one fragment.")
        else:
            if return_new_atom_indices  == False:
                output.append(Chem.MolToSmiles(combined_mol))
            else:
                output.append(combined_mol)
                new_atom_indices.append(list(range(mol_empty.GetNumAtoms(), combined_mol.GetNumAtoms())))
    return output if not return_new_atom_indices else list(zip(output, new_atom_indices))


def diffLinker_fragment_replacement(mol, toDelete, folder = "masking", return_new_atom_indices = False):
    """
    Delete specified atoms from a molecule and generate missing parts using DiffLinker.
    Parameters:
    ----------
    mol : rdkit.Chem.Mol
        The input molecule to be modified.
    toDelete : list of int
        A list of atom indices in the molecule that should be changed.
    folder : str, optional, default="masking"
        The name of the folder where intermediate and output files will be stored.
    return_new_atom_indices : bool, optional, default=False
        If True, returns the new atom indices of the generated molecules.
    Returns:
    -------
    list of str
        A list of SMILES strings representing the modified molecules generated 
        by combining the original molecule with the new parts.
    """
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    prepare_folders(folder, project_path)

    if mol.GetNumConformers() == 0:
        res = AllChem.EmbedMolecule(mol)
        if res == 0:
            AllChem.UFFOptimizeMolecule(mol)
        else:
            return []

    Chem.Kekulize(mol, clearAromaticFlags=True)
    # Create a bonds dictionary
    bonds_dict = create_graph_dictionary(mol)
    # Find connected components in the subset of atoms to delete
    connected_components = find_connections(toDelete, bonds_dict)
    # Map indices of atoms in the original molecule to the new indices after deletion
    old_to_new_index_map = map_old_to_new_indices(mol, toDelete)
    trueSmile = Chem.MolToSmiles(mol)

    # Generate missing fragments asynchronously
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, component in enumerate(connected_components):
            # Submit each component's processing as a separate task
            futures.append(executor.submit(process_component, idx, component, mol, folder, bonds_dict, project_path))
        # Wait for all tasks to complete before continuing
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in fragment generation: {e}")
    num_variants = 5 
    all_fragments = [[] for _ in range(len(connected_components))]

    for idx, component in enumerate(connected_components):
        for i in range(num_variants):
            xyz_file_path = get_project_path("data_diff", folder, f"{idx}_diff_output", f"output_{i}_fragments_{idx}_.xyz", base=project_path)
            mol_from_xyz = Chem.MolFromXYZFile(xyz_file_path)
            if mol_from_xyz is None:
                continue
            try:
                Chem.SanitizeMol(mol_from_xyz)
            except Exception:
                continue
            all_fragments[idx].append(mol_from_xyz)

    fragment_sizes = [len(fragments) for fragments in all_fragments]
    # Total number of combinations
    total_combinations = prod(fragment_sizes)

    if total_combinations <= 50:
        fragment_combinations = list(product(*all_fragments))
    else:
        sampled_indices = random.sample(range(total_combinations), 50)
        fragment_combinations = [flat_index_to_tuple(i, fragment_sizes) for i in sampled_indices]
        fragment_combinations = [tuple(all_fragments[j][idx[j]] for j in range(len(all_fragments))) for idx in fragment_combinations]
    
    # Combine and write out molecules
    for combo_idx, fragment_combo in enumerate(fragment_combinations):
        original_mol = atomDeleter(mol, toDelete)

        for frag, component in zip(fragment_combo, connected_components):
            num_atoms = frag.GetNumAtoms()
            atoms_to_keep = list(range(num_atoms - len(component), num_atoms))
            atoms_to_delete = [j for j in range(num_atoms) if j not in atoms_to_keep]
            emol = Chem.EditableMol(frag)
            for j in sorted(atoms_to_delete, reverse=True):
                emol.RemoveAtom(j)
            original_mol = Chem.CombineMols(original_mol, emol.GetMol())

        combined_xyz_path = get_project_path("data_diff", folder, "combined", f"combined_molecule_{combo_idx}.xyz", base=project_path)
        try:
            Chem.SanitizeMol(original_mol)
        except Exception:
            continue
        Chem.MolToXYZFile(original_mol, combined_xyz_path)
        load_rdkit_molecule(combined_xyz_path, get_project_path("data_diff", folder, "gen", f"gen_{combo_idx}.sdf", base=project_path), trueSmile)
    
    output_molecules = []
    # Fix bonds in generated molecules
    for i in range(len(fragment_combinations)):
        try:
            gen_sdf_path = get_project_path("data_diff", folder, "gen", f"gen_{i}.sdf", base=project_path)
            generated_mol = Chem.MolFromMolFile(gen_sdf_path)

            if generated_mol is None:
                print(f"Warning: Failed to load generated molecule {i}")
                continue

            emol = Chem.EditableMol(generated_mol)
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                if atom1 in toDelete or atom2 in toDelete:
                    continue

                atom1_in_generated = old_to_new_index_map.get(atom1)
                atom2_in_generated = old_to_new_index_map.get(atom2)

                if atom1_in_generated is None or atom2_in_generated is None:
                    continue

                temp_mol = emol.GetMol()
                if temp_mol.GetBondBetweenAtoms(atom1_in_generated, atom2_in_generated) is not None:
                    emol.RemoveBond(atom1_in_generated, atom2_in_generated)

                emol.AddBond(atom1_in_generated, atom2_in_generated, bond_type)

            if emol.GetMol().GetNumAtoms() != mol.GetNumAtoms():
                continue
            num_fragments = len(Chem.GetMolFrags(emol.GetMol()))
            if num_fragments > 1:
                continue
            output_mol = emol.GetMol()
            try:
                Chem.SanitizeMol(output_mol)
            except Exception as e:
                continue
            output_sdf_path = get_project_path("data_diff", folder, "output", f"gen_{i}.sdf", base=project_path)
            with Chem.SDWriter(output_sdf_path) as writer:
                writer.write(output_mol)
            if return_new_atom_indices == False:
                output_molecules.append(Chem.MolToSmiles(output_mol))
            else:
                new_atom_indices = list(range(mol.GetNumAtoms() - len(toDelete), output_mol.GetNumAtoms()))
                output_molecules.append((output_mol, new_atom_indices))
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
    return output_molecules
    
def flat_index_to_tuple(idx, sizes):
    result = []
    for size in reversed(sizes):
        result.append(idx % size)
        idx //= size
    return tuple(reversed(result))