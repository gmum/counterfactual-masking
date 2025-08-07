from rdkit import Chem
import time
import requests
from rdkit.Chem import BRICS

def get_substructures(smiles):
    """
    Decompose a molecule into BRICS substructures.
    """
    unique_substructures = set()
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fragments = BRICS.BRICSDecompose(mol)
        for frag in fragments:
            unique_substructures.add(frag)
    return unique_substructures

def remove_attachment_points(smiles):
    """
    Remove BRICS attachment points (dummy atoms) from a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)  
    if mol:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atoms (*, [13*], [15*])
                atom.SetAtomicNum(1)  # Convert to hydrogen
            atom.SetIsotope(0)  # Remove isotopic labels
        return Chem.MolToSmiles(mol, isomericSmiles=False)  
    return None

def fetch_superstructure_data(smiles_string):
    """
    Query PubChem for compounds containing a given substructure.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    endpoint = f"/compound/substructure/smiles/{smiles_string}/JSON?MaxRecords=30"
    response = requests.get(base_url + endpoint)

    data = None

    if response.status_code == 202:  
        list_key = response.json()["Waiting"]["ListKey"]

        poll_url = f"{base_url}/compound/listkey/{list_key}/JSON"
    
        while True:
            time.sleep(3)  
            poll_response = requests.get(poll_url)

            if poll_response.status_code == 200: 
                data = poll_response.json()
                # print("Results retrieved successfully!")
                break
            elif poll_response.status_code == 202:
                print("Still processing, waiting...")
            else:
                print(f"Error: {poll_response.status_code}, {poll_response.text}")
                break
    elif response.status_code == 200: 
        data = response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")  

    if data is None:
        print("No data returned from PubChem.")
        return []
    
    smiles_list = []
    for compound in data.get('PC_Compounds', []):
        props = compound.get('props', [])
        for prop in props:
            urn = prop.get('urn', {})
            if urn.get('label') == 'SMILES' and urn.get('name') == 'Absolute':
                smiles_value = prop.get('value', {}).get('sval')
                if smiles_value:
                    smiles_list.append(smiles_value)
                break

    return smiles_list