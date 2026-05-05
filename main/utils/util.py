from rdkit import Chem
from rdkit.Chem import BRICS

def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor']:
        return 1
    elif dataset == 'pcba':
        return 92
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset in ['esol', 'freesolv', 'lipophilicity']:
        return 1
    
    raise ValueError('Invalid dataset name.')

def get_clique_mol(mol, atoms):
    if len(atoms) == 0: 
        return None
    try:
        return Chem.MolFragmentToSmiles(mol, atoms, isomericSmiles=True, canonical=True)
    except: 
        return None

def decompose_to_cliques(mol):
    res_mol = Chem.RWMol(mol)
    for bond in list(BRICS.FindBRICSBonds(mol)):
        u, v = bond[0]
        res_mol.RemoveBond(u, v)
    initial_cliques = [list(frag) for frag in Chem.GetMolFrags(res_mol, asMols=False)]
    ssr_mol = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
    final_cliques = []
    for c in initial_cliques:
        found_rings = [r for r in ssr_mol if set(r).issubset(set(c))]
        if len(found_rings) > 0:
            for ring in found_rings: 
                final_cliques.append(ring)
            all_ring_atoms = set().union(*found_rings)
            remaining = list(set(c) - all_ring_atoms)
            if len(remaining) > 0: 
                final_cliques.append(remaining)
        else: 
            final_cliques.append(c)
    return final_cliques
