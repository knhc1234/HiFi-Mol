import os
from itertools import repeat

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from torch_geometric.data import (Data, InMemoryDataset)
import json
import pyarrow.parquet as pq
from utils.util import get_clique_mol, decompose_to_cliques

allowable_features = {
    # Atom
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_is_aromatic_list':      [False, True],      
    'possible_is_in_ring_list':       [False, True],       
    'possible_num_radical_e_list':    [0, 1, 2, 3, 4],     
    
    # Bond Features
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_stereo_list': [                        
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ],
    'possible_is_conjugated_list': [False, True]          
}

def get_hierarchical_info_dense(mol, final_cliques):
    num_atoms = mol.GetNumAtoms()
    
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
    
    return torch.tensor(hier_edges, dtype=torch.long).t().contiguous(), torch.tensor(hier_attr, dtype=torch.long)

def mol_to_graph_hierarchical_dense(mol, vocab):
    cliques = decompose_to_cliques(mol)
    num_atoms = mol.GetNumAtoms()
    num_frags = len(cliques)
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
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
    x = torch.tensor(atom_features, dtype=torch.long)
    
    hier_edge_index, hier_edge_attr = get_hierarchical_info_dense(mol, cliques)
    
    frag_ids = [0] * num_atoms
    for c in cliques:
        frag_ids.append(vocab.get(get_clique_mol(mol, c), 1)) # 1: UNK
    frag_ids.append(2) # 2: GLOBAL
    
    node_type = torch.zeros(num_atoms + num_frags + 1, dtype=torch.long)
    node_type[num_atoms : num_atoms + num_frags] = 1 # Frag
    node_type[num_atoms + num_frags] = 2             # Global

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
    
    bond_edge_index = torch.tensor(bond_edges, dtype=torch.long).t()
    bond_edge_attr = torch.tensor(bond_attr, dtype=torch.long)

    full_edge_index = torch.cat([bond_edge_index, hier_edge_index], dim=1)
    full_edge_attr = torch.cat([bond_edge_attr, hier_edge_attr], dim=0)
    
    data = Data(x=x, edge_index=full_edge_index, edge_attr=full_edge_attr)
    data.node_type = node_type
    data.frag_ids = torch.tensor(frag_ids, dtype=torch.long)
    data.num_atoms = torch.tensor([num_atoms])
    data.num_frags = torch.tensor([num_frags])
    return data

class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, dataset='bbbp', transform=None,
                 pre_transform=None, pre_filter=None, empty=False):

        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        if self.dataset == 'davis':
            file_name_list = ['davis']
        elif self.dataset == 'kiba':
            file_name_list = ['kiba']
        else:
            file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        vocab_path = "../datasets/pcqm4m-v2_dense/pcqm_vocab.json"
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        print(f"Loaded vocab from {vocab_path}")
        
        # Get Fingerprint Embeddings
        parquet_path = '../datasets/downstream_fingerprint_embeddings.parquet'
        
        if self.dataset == 'toxcast':
            target_id = 'toxcast_data'
        elif self.dataset == 'hiv':
            target_id = 'HIV'
        elif self.dataset == 'bbbp':
            target_id = 'BBBP'
        elif self.dataset == 'lipophilicity':
            target_id = 'lipo'
        else:
            target_id = self.dataset
        
        table = pq.read_table(
            parquet_path,
            columns=['SMILES', 'LIBRARY_ID', 'embeddings'],
            filters=[('LIBRARY_ID', '==', target_id)],
            thrift_string_size_limit=2_000_000_000,
            thrift_container_size_limit=2_000_000_000
        )
        
        filtered_parquet = table.to_pandas()
        emb_map = {}
        for _, row in filtered_parquet.iterrows():
            s_raw = row['SMILES'].strip()
            m_tmp = Chem.MolFromSmiles(s_raw)

            if m_tmp is not None:
                s_canon = Chem.MolToSmiles(m_tmp, isomericSmiles=True).strip()
                emb_map[s_canon] = row['embeddings']
        
        def shared_extractor(smiles_list, rdkit_mol_objs, labels, emb_map=None):
            data_list = []
            data_smiles = []
            final_embs = []
            
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=1)
            
            for i, mol in enumerate(rdkit_mol_objs):
                if mol is None: continue
                canon_s = Chem.MolToSmiles(mol, isomericSmiles=True).strip()
                if canon_s not in emb_map: continue
                
                data = mol_to_graph_hierarchical_dense(mol, vocab)
                data.id = torch.tensor([len(data_list)])
                data.y = torch.tensor(labels[i])
                
                data_list.append(data)
                data_smiles.append(smiles_list[i])
                final_embs.append(emb_map[canon_s])
            return data_list, data_smiles, final_embs
        
        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)
        
        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            data_list, data_smiles_list, final_embs = shared_extractor(
                smiles_list, rdkit_mol_objs, labels, emb_map=emb_map)

        else:
            raise ValueError('Dataset {} not included.'.format(self.dataset))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        if final_embs:
            np.save(os.path.join(self.processed_dir, 'fp_embeddings.npy'), np.stack(final_embs))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return

def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values

def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values

def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values

def _load_muv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_sider_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_toxcast_dataset(input_path):

    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values

def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_freesolv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values