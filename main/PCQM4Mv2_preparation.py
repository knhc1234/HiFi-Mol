import os
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from torch_geometric.data import Data
from collections import Counter
from datasets import allowable_features
from utils.util import get_clique_mol, decompose_to_cliques
RDLogger.DisableLog('rdApp.*')

def get_hierarchical_info_3D(mol, final_cliques):
    num_atoms = mol.GetNumAtoms()
    conformer = mol.GetConformer()
    atom_pos = torch.tensor(conformer.GetPositions(), dtype=torch.float)
    
    num_frags = len(final_cliques)
    frag_offset = num_atoms
    global_node_idx = num_atoms + num_frags
    
    hier_edges, hier_attr = [], []
    atom_to_frags = [[] for _ in range(num_atoms)]
    
    for f_idx, atom_indices in enumerate(final_cliques):
        f_node_idx = frag_offset + f_idx
        for a_idx in atom_indices:
            atom_to_frags[a_idx].append(f_idx)
            hier_edges.extend([[a_idx, f_node_idx], [f_node_idx, a_idx]])
            hier_attr.extend([[6, 0, 0, 0], [7, 0, 0, 0]])
        hier_edges.extend([[f_node_idx, global_node_idx], [global_node_idx, f_node_idx]])
        hier_attr.extend([[8, 0, 0, 0], [9, 0, 0, 0]])

    added_frag_edges = set()
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for f_u in atom_to_frags[u]:
            for f_v in atom_to_frags[v]:
                if f_u != f_v:
                    edge_pair = tuple(sorted((f_u, f_v)))
                    if edge_pair not in added_frag_edges:
                        hier_edges.extend([[frag_offset+f_u, frag_offset+f_v], [frag_offset+f_v, frag_offset+f_u]])
                        hier_attr.extend([[10, 0, 0, 0], [10, 0, 0, 0]])
                        added_frag_edges.add(edge_pair)
    
    total_nodes = global_node_idx + 1
    pos = torch.zeros((total_nodes, 3), dtype=torch.float)
    pos[:num_atoms] = atom_pos
    for f_idx, atom_indices in enumerate(final_cliques):
        pos[frag_offset + f_idx] = atom_pos[atom_indices].mean(dim=0)
    pos[global_node_idx] = atom_pos.mean(dim=0)
    
    return torch.tensor(hier_edges, dtype=torch.long).t().contiguous(), torch.tensor(hier_attr, dtype=torch.long), pos

def mol_to_graph_hierarchical_pcq(mol, vocab):
    if mol is None: 
        return None
    try:
        cliques = decompose_to_cliques(mol)
        num_atoms = mol.GetNumAtoms()
        num_frags = len(cliques)
        
        atom_x = []
        for atom in mol.GetAtoms():
            atom_x.append([
                allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),
                allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),
                allowable_features['possible_formal_charge_list'].index(atom.GetFormalCharge()),
                allowable_features['possible_hybridization_list'].index(atom.GetHybridization()),
                allowable_features['possible_numH_list'].index(atom.GetTotalNumHs()),
                allowable_features['possible_implicit_valence_list'].index(atom.GetImplicitValence()),
                allowable_features['possible_degree_list'].index(atom.GetDegree()),
                allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()), # 추가
                allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),      # 추가
                allowable_features['possible_num_radical_e_list'].index(atom.GetNumRadicalElectrons()), # 추가
            ])
        x = torch.tensor(atom_x, dtype=torch.long)  # [num_atoms, 10]

        hier_edge_index, hier_edge_attr, pos = get_hierarchical_info_3D(mol, cliques)

        frag_ids = [0] * num_atoms
        for c in cliques:
            frag_ids.append(vocab.get(get_clique_mol(mol, c), 1))
        frag_ids.append(2)

        node_type = torch.zeros(x.size(0) + num_frags + 1, dtype=torch.long)
        node_type[num_atoms : num_atoms + num_frags] = 1
        node_type[num_atoms + num_frags] = 2
        
        frag_virtual_x = torch.tensor([120, 5, 11, 7, 9, 7, 11, 2, 2, 5]).repeat(num_frags, 1)
        global_virtual_x = torch.tensor([121, 6, 12, 8, 10, 8, 12, 3, 3, 6]).reshape(1, 10)
        
        extra_x = torch.cat([frag_virtual_x, global_virtual_x], dim=0)
        
        x = torch.cat([x, extra_x], dim=0)

        bond_edges, bond_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            attr = [allowable_features['possible_bonds'].index(bond.GetBondType()),
                    allowable_features['possible_bond_dirs'].index(bond.GetBondDir()),
                    allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),   # 추가
                    allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated())] # 추가
            bond_edges.extend([[i, j], [j, i]])
            bond_attr.extend([attr, attr])

        full_edge_index = torch.cat([torch.tensor(bond_edges, dtype=torch.long).t(), hier_edge_index], dim=1)
        full_edge_attr = torch.cat([torch.tensor(bond_attr, dtype=torch.long), hier_edge_attr], dim=0)

        data = Data(x=x, edge_index=full_edge_index, edge_attr=full_edge_attr)
        data.positions = pos
        data.node_type = node_type
        data.frag_ids = torch.tensor(frag_ids, dtype=torch.long)
        data.num_atoms = torch.tensor([num_atoms])
        data.num_frags = torch.tensor([num_frags])
        
        return data
    except: 
        return None

# Main Process
def process_pcq_full(sdf_path, output_dir, block_size=100000, min_freq=10):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    vocab_path = os.path.join(output_dir, "pcqm_vocab.json")

    # [STEP 1] Fragment Vocabulary Generation
    step = 0
    if not os.path.exists(vocab_path):
        print("--- Step 1: Building Vocabulary (Scanning entire SDF) ---")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        frag_counter = Counter()
        for mol in tqdm(suppl, total=3800000, desc="Vocab Building"):
            if mol is None: 
                continue
            cliques = decompose_to_cliques(mol)
            for c in cliques:
                s = get_clique_mol(mol, c)
                if s: 
                    frag_counter[s] += 1
            step += 1
            if step >= 120000:
                break
        
        vocab = {"<PAD>": 0, "<UNK>": 1, "<GLOBAL>": 2}
        for s, count in frag_counter.most_common():
            if count >= min_freq: 
                vocab[s] = len(vocab)
        with open(vocab_path, 'w') as f: 
            json.dump(vocab, f)
        print(f"Vocab saved. Total size: {len(vocab)}")
    else:
        with open(vocab_path, 'r') as f: vocab = json.load(f)
        print(f"Loaded existing vocab from {vocab_path}")

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    print(f"--- Vocab Loaded from {vocab_path} (Total size: {len(vocab)}) ---")
    
    print("--- Step 2: Converting to Hierarchical Graphs & Saving in Blocks ---")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    current_block = []
    block_id = 0
    
    for i, mol in enumerate(tqdm(suppl, total=3800000, desc="Graph Processing")):
        data = mol_to_graph_hierarchical_pcq(mol, vocab)
        if data is not None:
            current_block.append(data)
        
        if len(current_block) >= block_size:
            save_path = os.path.join(output_dir, f'pcq_block_{block_id}.pt')
            torch.save(current_block, save_path)
            print(f" Block {block_id} saved ({len(current_block)} samples)")
            block_id += 1
            current_block = [] 
            break
        
    if current_block:
        torch.save(current_block, os.path.join(output_dir, f'pcq_block_{block_id}.pt'))
    
    print("All processes complete!")

if __name__ == '__main__':
    SDF_FILE_PATH = "../datasets/pcqm4m-v2-train.sdf" 
    OUTPUT_DIRECTORY = "../datasets/pcqm4m-v2_dense"
    
    process_pcq_full(SDF_FILE_PATH, OUTPUT_DIRECTORY, block_size=100000)