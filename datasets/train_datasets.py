import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

class HierarchicalData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'fp_embeddings':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, molecule_dataset, fp_embeddings_mmap, mask_ratio_atom=0.25, mask_ratio_frag=0.15, mask_token=18907):
        self.molecule_dataset = molecule_dataset
        self.fp_embeddings = fp_embeddings_mmap
        self.mask_ratio_atom = mask_ratio_atom
        self.mask_ratio_frag = mask_ratio_frag
        self.frag_mask_token = mask_token
        self.atom_mask_token = 119

    def __len__(self):
        return len(self.molecule_dataset)

    def _get_augmented_view(self, idx):
        data = self.molecule_dataset[idx].clone()
        num_atoms = data.num_atoms.item()
        num_frags = data.num_frags.item()
        
        x_target = data.x[:num_atoms, 0].clone()    # Atom Type Label
        frag_target = data.frag_ids[num_atoms : num_atoms + num_frags].clone() 
        pos_target = data.positions[:num_atoms].clone() if hasattr(data, 'positions') else None
        
        m_g_mask = torch.zeros(num_atoms, dtype=torch.bool)
        m_f_mask = torch.zeros(num_frags, dtype=torch.bool)
        
        # Fragment-aware masking
        mask_num_f = max(1, int(num_frags * self.mask_ratio_frag))
        f_indices = torch.randperm(num_frags)[:mask_num_f]
        
        total_masked_atoms = []
        for f_idx in f_indices:
            # (1) Fragment node masking
            f_node_idx = num_atoms + f_idx
            data.frag_ids[f_node_idx] = self.frag_mask_token
            m_f_mask[f_idx] = True
            
            # (2) Track atoms belonging to each fragment and masking
            mask = (data.edge_index[1] == f_node_idx)
            connected_atoms = data.edge_index[0][mask]
            connected_atoms = connected_atoms[connected_atoms < num_atoms]
        
            data.x[connected_atoms, 0] = self.atom_mask_token
            m_g_mask[connected_atoms] = True
            total_masked_atoms.append(connected_atoms)
        
        # Random atom masking (additional)
        current_masked_ratio = m_g_mask.sum().item() / num_atoms
        
        if current_masked_ratio < self.mask_ratio_atom:
            target_num_a = int(num_atoms * self.mask_ratio_atom)
            additional_num_a = target_num_a - m_g_mask.sum().item()
            
            # Sample from atoms that have not yet been masked
            unmasked_indices = torch.where(~m_g_mask)[0]
            if len(unmasked_indices) > 0:
                actual_add_num = min(len(unmasked_indices), int(additional_num_a))
                add_indices = unmasked_indices[torch.randperm(len(unmasked_indices))[:actual_add_num]]
                
                data.x[add_indices, 0] = self.atom_mask_token
                m_g_mask[add_indices] = True
                total_masked_atoms.append(add_indices)
        
        ATOM_MASK_LIST = [119, 7, 11, 7, 9, 7, 11, 2, 2, 5]
        EDGE_MASK_LIST = [5, 3, 6, 2]
        for i in range(10):
            data.x[:num_atoms][m_g_mask, i] = ATOM_MASK_LIST[i]
        
        # Preserve subgraph structural information (Internal Edges)
        all_masked_indices = torch.where(m_g_mask)[0]
        masked_edge_index, masked_edge_attr = subgraph(
            all_masked_indices, 
            data.edge_index, 
            edge_attr=data.edge_attr, 
            relabel_nodes=False
        )
        
        # Remove edge attributes of masked regions from input data
        # Mask edge attributes if at least one endpoint node is masked
        edge_in_masked = torch.isin(data.edge_index[0], all_masked_indices)
        edge_out_masked = torch.isin(data.edge_index[1], all_masked_indices)
        
        # Edge attribute mask condition: True if either endpoint is masked
        attr_mask_condition = edge_in_masked | edge_out_masked
        
        masked_edge_index = data.edge_index[:, attr_mask_condition]
        masked_edge_attr = data.edge_attr[attr_mask_condition].clone()
        
        # Mask edge attributes in input data (hiding ground truth)
        for i in range(4):
            data.edge_attr[attr_mask_condition, i] = EDGE_MASK_LIST[i]
        
        # Load and transform additional data(Future Works)
        if self.fp_embeddings:
            fp_emb = torch.from_numpy(self.fp_embeddings[idx].copy()).float().unsqueeze(0)
        else:
            fp_emb = None

        return HierarchicalData(
            x = data.x,
            edge_index = data.edge_index,   # Merged edges
            edge_attr = data.edge_attr,     # Atom bond attributes (hierarchical edges padded with 0 or ignored)
            node_type = data.node_type,     # 0: Atom, 1: Frag, 2: Global          
            pos = data.positions,           # coordinates
            fp_embeddings= fp_emb,
            # MAE Targets
            x_target=x_target,         # [N_atoms] Ground truth atom types
            frag_target = frag_target, # [N_frags] Fragment ground truth
            pos_target = pos_target,   # [N_atoms, 3] 3D ground truth (atom-level)
            edge_target_index = masked_edge_index,  # Actual connectivity of masked subgraph
            edge_target_attr = masked_edge_attr,    # Actual bond types of masked subgraph
            m_g_mask = m_g_mask,       # Atom mask positions
            m_f_mask = m_f_mask,       # Fragment mask positions
            num_atoms = data.num_atoms,
            num_frags = data.num_frags,
            frag_ids = data.frag_ids
        )

    def __getitem__(self, idx):
        # 1. original data
        view1 = self._get_augmented_view(idx)
        view2 = self._get_augmented_view(idx)
        return view1, view2

class HiFi_Mol_Downstream_Dataset(torch.utils.data.Dataset):
    def __init__(self, molecule_dataset, fp_embeddings_mmap):
        self.molecule_dataset = molecule_dataset
        self.fp_embeddings = fp_embeddings_mmap

    def __len__(self):
        return len(self.molecule_dataset)

    def __getitem__(self, idx):
        # original data
        data = self.molecule_dataset[idx].clone()
        fp_emb = torch.from_numpy(self.fp_embeddings[idx].copy()).float().unsqueeze(0)
        
        new_data = HierarchicalData(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            node_type=data.node_type,
            fp_embeddings=fp_emb,
            
            y=data.y,
            num_atoms=data.num_atoms,
            num_frags=data.num_frags,
            frag_ids = data.frag_ids,
            orig_idx=torch.tensor([idx])
        )
        
        return new_data