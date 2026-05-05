import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add, scatter_mean, scatter_max

num_atom_type = 120  # including the extra mask tokens(Mask: 119, Frag: 120, Global: 121)
num_formal_charge = 12  # 11
num_chirality_tag = 8   # (Self-loop: 4, Frag: 5, Global: 6, Mask: 7)
num_hybridization = 8  # 7   
num_H = 10          # 9
num_imp_val = 8    # 7
num_degree = 12     # 11
num_is_aromatic = 3  # 2
num_is_in_ring  = 3  # 2
num_radical_e = 6    # 5

num_bond_type = 11  # including aromatic and self-loop edge, and extra masked tokens 
num_bond_direction = 4
num_bond_stereo = 7
num_conjugated = 3

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        self.aggr = aggr
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                                 nn.ReLU(),
                                 nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_stereo, emb_dim)
        self.edge_embedding4 = nn.Embedding(num_conjugated, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding4.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1]) + \
                          self.edge_embedding3(edge_attr[:, 2]) + \
                          self.edge_embedding4(edge_attr[:, 3])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.aggr = aggr
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_stereo, emb_dim)
        self.edge_embedding4 = nn.Embedding(num_conjugated, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding4.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1]) + \
                          self.edge_embedding3(edge_attr[:, 2]) + \
                          self.edge_embedding4(edge_attr[:, 3])

        norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_stereo, heads * emb_dim)
        self.edge_embedding4 = nn.Embedding(num_conjugated, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding4.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1]) + \
                          self.edge_embedding3(edge_attr[:, 2]) + \
                          self.edge_embedding4(edge_attr[:, 3])

        x = self.weight_linear(x)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)
        
    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()
        self.aggr = aggr

        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_stereo, emb_dim)
        self.edge_embedding4 = nn.Embedding(num_conjugated, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding4.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1]) + \
                          self.edge_embedding3(edge_attr[:, 2]) + \
                          self.edge_embedding4(edge_attr[:, 3])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)

class HierarchicalGNN(nn.Module):
    def __init__(self, num_layer, emb_dim, vocab_size, JK="last", drop_ratio=0., gnn_type="gcn"):
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        super(HierarchicalGNN, self).__init__()
        
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)        # atom_num
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)    # chiral
        self.x_embedding3 = nn.Embedding(num_formal_charge, emb_dim)    # formal_charge
        self.x_embedding4 = nn.Embedding(num_hybridization, emb_dim)    # hybridization
        self.x_embedding5 = nn.Embedding(num_H, emb_dim)                # numH
        self.x_embedding6 = nn.Embedding(num_imp_val, emb_dim)          # implicit_valence
        self.x_embedding7 = nn.Embedding(num_degree, emb_dim)           # degree
        self.x_embedding8 = nn.Embedding(num_is_aromatic, emb_dim)      # is_aromatic
        self.x_embedding9 = nn.Embedding(num_is_in_ring, emb_dim)       # is_in_ring
        self.x_embedding10 = nn.Embedding(num_radical_e, emb_dim)       # num_radical_e
        
        # Fragment Embeding
        self.frag_embedding = nn.Embedding(vocab_size + 1, self.emb_dim)
        
        # 0: Atom, 1: Fragment, 2: Global
        self.type_embedding = nn.Embedding(3, self.emb_dim) 
        
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        nn.init.xavier_uniform_(self.x_embedding9.weight.data)
        nn.init.xavier_uniform_(self.x_embedding10.weight.data)
        nn.init.xavier_uniform_(self.frag_embedding.weight.data)
        nn.init.xavier_uniform_(self.type_embedding.weight.data)

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            # Atom-Atom / Atom-Motif Message-Passing GNN
            if gnn_type == "gin":
                self.gnns.append(GINConv(self.emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(self.emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(self.emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(self.emb_dim))    
            
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

    def forward(self, data, global_init=None):
        # x: [Total_Nodes, 10], edge_index: [2, Total_Edges], node_type: [Total_Nodes]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_type = data.node_type     
        batch = data.batch              # [Total_Nodes_in_Batch]
        
        atom_mask = (node_type == 0)
        frag_mask = (node_type == 1)
        global_mask = (node_type == 2)
        
        h = torch.zeros((x.size(0), self.emb_dim), device=x.device)
        
        h[atom_mask] = (self.x_embedding1(x[atom_mask, 0]) + 
                        self.x_embedding2(x[atom_mask, 1]) + 
                        self.x_embedding3(x[atom_mask, 2]) + 
                        self.x_embedding4(x[atom_mask, 3]) + 
                        self.x_embedding5(x[atom_mask, 4]) + 
                        self.x_embedding6(x[atom_mask, 5]) + 
                        self.x_embedding7(x[atom_mask, 6]) + 
                        self.x_embedding8(x[atom_mask, 7]) + 
                        self.x_embedding9(x[atom_mask, 8]) + 
                        self.x_embedding10(x[atom_mask, 9]))
        h[frag_mask] = self.frag_embedding(data.frag_ids[frag_mask])
        
        if global_init is not None:
            h[global_mask] = global_init
        else:
            h[global_mask] = scatter_mean(h[atom_mask], batch[atom_mask], dim=0)
        
        h = h + self.type_embedding(node_type)
        
        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # 3. JK Connection
        if self.JK == "last":
            node_rep = h_list[-1]
        elif self.JK == "concat":
            node_rep = torch.cat(h_list, dim=1)
        
        return node_rep

class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)        # atom_num
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)    # chiral
        self.x_embedding3 = nn.Embedding(num_formal_charge, emb_dim)    # formal_charge
        self.x_embedding4 = nn.Embedding(num_hybridization, emb_dim)    # hybridization
        self.x_embedding5 = nn.Embedding(num_H, emb_dim)                # numH
        self.x_embedding6 = nn.Embedding(num_imp_val, emb_dim)          # implicit_valence
        self.x_embedding7 = nn.Embedding(num_degree, emb_dim)           # degree
        self.x_embedding8 = nn.Embedding(num_is_aromatic, emb_dim)      # is_aromatic
        self.x_embedding9 = nn.Embedding(num_is_in_ring, emb_dim)       # is_in_ring
        self.x_embedding10 = nn.Embedding(num_radical_e, emb_dim)       # num_radical_e
    
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        nn.init.xavier_uniform_(self.x_embedding9.weight.data)
        nn.init.xavier_uniform_(self.x_embedding10.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = (self.x_embedding1(x[:, 0]) + 
            self.x_embedding2(x[:, 1]) + 
            self.x_embedding3(x[:, 2]) + 
            self.x_embedding4(x[:, 3]) + 
            self.x_embedding5(x[:, 4]) + 
            self.x_embedding6(x[:, 5]) + 
            self.x_embedding7(x[:, 6]) + 
            self.x_embedding8(x[:, 7]) + 
            self.x_embedding9(x[:, 8]) + 
            self.x_embedding10(x[:, 9]))
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation

from torch_geometric.utils import to_dense_batch

class DistancePredictor(nn.Module):
    def __init__(self, d, m = 32):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d, m),
            nn.ReLU(),
            nn.LayerNorm(m)
        )
        self.final_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(m * m, m),
            nn.ReLU(),
            nn.Linear(m, 1)
        )
    def forward(self, h_i, h_j):
        m_i = self.projector(h_i) # [N, m]
        m_j = self.projector(h_j) # [N, m]
        
        # 3. Outer Product (Batch-wise)
        relation_matrix = torch.bmm(m_i.unsqueeze(2), m_j.unsqueeze(1)) # [N, m, m]
        
        # 4. Flatten + Linear
        dist = self.final_head(relation_matrix) # [N, 1]
        return dist.squeeze()

def info_nce_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss

class HierarchicalPretrain(nn.Module):
    def __init__(self, args, molecule_model, vocab_size):
        super().__init__()
        self.molecule_model = molecule_model    # HierarchicalGNN
        self.emb_dim = args.emb_dim
        
        self.proj_global = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, 128) 
        )
        
        # 1. Reconstruction Heads
        self.atom_pred_head = nn.Linear(self.emb_dim, 119) 
        self.frag_pred_head = nn.Linear(self.emb_dim, vocab_size)
        self.dist_predictor = DistancePredictor(self.emb_dim, m=32)
        
        # Bond Type Prediction (4type + No Bond)
        self.edge_pred_head = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, 5) 
        )
        
        self.log_vars = nn.Parameter(torch.zeros(5))    # [CL, Atom_Type, 3D_Pos]
        self._init_weights()
    
    def _init_weights(self):
        new_components = [
            self.atom_pred_head, 
            self.dist_predictor, 
            self.frag_pred_head, 
            self.edge_pred_head,
        ]
        
        for component in new_components:
            for m in component.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _calculate_all_pairs_dist_loss(self, h_a, batch):
        device = h_a.device
        atom_indices = torch.where(batch.node_type == 0)[0]
        batch_atoms = batch.batch[atom_indices]
        pos_target = batch.pos_target # [Total_Atoms, 3]
        
        # 1. Real Bond (Short-range)
        edge_idx = batch.edge_target_index
        real_bond_mask = batch.edge_target_attr[:, 0] < 5
        edge_idx_bonds = edge_idx[:, real_bond_mask]
        
        # 2. Random Sampling (Long-range & Non-bonded)
        num_atoms = h_a.size(0)
        idx_i = torch.randint(0, num_atoms, (num_atoms * 2,), device=device)
        idx_j = torch.randint(0, num_atoms, (num_atoms * 2,), device=device)
        
        # Filtering (Same-molecule mask) & Integrate (1 & 2)
        same_mol_mask = (batch_atoms[idx_i] == batch_atoms[idx_j]) & (idx_i != idx_j)
        idx_i, idx_j = idx_i[same_mol_mask], idx_j[same_mol_mask]
        
        atom_map = torch.full((batch.x.size(0),), -1, dtype=torch.long, device=device)
        atom_map[atom_indices] = torch.arange(len(atom_indices), device=device)
        
        bond_i, bond_j = atom_map[edge_idx_bonds[0]], atom_map[edge_idx_bonds[1]]
        
        final_idx_i = torch.cat([bond_i, idx_i])
        final_idx_j = torch.cat([bond_j, idx_j])
        
        if final_idx_i.size(0) == 0:
            return torch.tensor(0.0, device=device)

        # 4. Distance Pred
        target_dist = torch.norm(pos_target[final_idx_i] - pos_target[final_idx_j], dim=-1)
        pred_dist = self.dist_predictor(h_a[final_idx_i], h_a[final_idx_j])
        
        return F.mse_loss(pred_dist, target_dist)

    def _calculate_bond_loss(self, h_a, batch):
        edge_idx = batch.edge_target_index
        edge_attr = batch.edge_target_attr
        
        real_bond_mask = edge_attr[:, 0] < 5
        if real_bond_mask.sum() == 0:
            return torch.tensor(0.0, device=h_a.device)
            
        edge_idx = edge_idx[:, real_bond_mask]
        edge_attr = edge_attr[real_bond_mask]
        
        row, col = edge_idx[0], edge_idx[1]
        atom_indices = torch.where(batch.node_type == 0)[0]
        atom_map = torch.full((batch.x.size(0),), -1, dtype=torch.long, device=h_a.device)
        atom_map[atom_indices] = torch.arange(len(atom_indices), device=h_a.device)
        
        # Calculate only one direction
        edge_mask_half = row < col
        if edge_mask_half.sum() > 0:
            row_m, col_m = atom_map[row[edge_mask_half]], atom_map[col[edge_mask_half]]
            edge_repr = torch.cat([h_a[row_m], h_a[col_m]], dim=-1)
            pred_bond = self.edge_pred_head(edge_repr)
            target_bond = edge_attr[edge_mask_half, 0].long()
            return F.cross_entropy(pred_bond, target_bond)
        return torch.tensor(0.0, device=h_a.device)

    def forward_batch(self, batch):
        # 1. Get Hierarchical Features
        node_rep = self.molecule_model(batch, global_init=None)
        
        atom_mask = (batch.node_type == 0)
        frag_mask = (batch.node_type == 1)
        global_mask = (batch.node_type == 2)
        
        h_a_hier = node_rep[atom_mask]
        h_f_hier = node_rep[frag_mask]
        h_g_hier = node_rep[global_mask]
        
        global_h = self.proj_global(h_g_hier)
        # Reconstruction Tasks
        # Atom Type
        loss_atom = F.cross_entropy(
            self.atom_pred_head(h_a_hier[batch.m_g_mask]), 
            batch.x_target[batch.m_g_mask]
        )
        # Frag Type
        loss_frag = F.cross_entropy(
            self.frag_pred_head(h_f_hier[batch.m_f_mask]), 
            batch.frag_target[batch.m_f_mask]
        )
        # Subgraph (Distance & Bond)
        loss_dist = self._calculate_all_pairs_dist_loss(h_a_hier, batch)
        loss_bond = self._calculate_bond_loss(h_a_hier, batch)
        
        return global_h, loss_atom, loss_frag, loss_dist, loss_bond

    def forward(self, batch1, batch2):
        # Get global-level feature and calculate total losses 
        h1, loss_a1, loss_f1, loss_d1, loss_b1 = self.forward_batch(batch1)
        h2, loss_a2, loss_f2, loss_d2, loss_b2 = self.forward_batch(batch2)
        
        loss_align = info_nce_loss(h1, h2)
        loss_atom = (loss_a1 + loss_a2)/2
        loss_frag = (loss_f1 + loss_f2)/2
        loss_dist = (loss_d1 + loss_d2)/2
        loss_bond = (loss_b1 + loss_b2)/2
        
        losses = [loss_align, loss_atom, loss_frag, loss_dist, loss_bond]
        total_loss = 0
        for i, l in enumerate(losses):  # Uncertainty weighting
            w = torch.exp(-self.log_vars[i])
            total_loss += w * l + self.log_vars[i]
        
        return total_loss, loss_align, loss_atom, loss_frag, loss_dist, loss_bond
    
class HiFi_Mol(nn.Module):
    def __init__(self, args, num_tasks, molecule_model, fp_dim=1024):
        super(HiFi_Mol, self).__init__()
        self.molecule_model = molecule_model  # HierarchicalGNN
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.dropout = args.dropout_ratio
        self.fp_dim = fp_dim
        self.n_heads = 8

        self.gnn_proj = nn.Sequential(
            nn.Linear(self.emb_dim * 3, self.fp_dim),
            nn.BatchNorm1d(self.fp_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fp_dim, 128)
        )

        self.fp_proj = nn.Sequential(           
            nn.Linear(self.fp_dim, self.fp_dim), 
            nn.BatchNorm1d(self.fp_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fp_dim, 128)
        )
        
        self.prediction_layer = nn.Sequential(
            nn.Linear(128 * 2, 128), # Hier_G + FP
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(128, num_tasks),
        )

    def from_pretrained(self, model_path, device):
        print(f"Loading pretrained backbones from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
            print(" > Found integrated model_state_dict. Extracting components...")
            hier_sd = {k.replace('molecule_model.', ''): v for k, v in sd.items() if k.startswith('molecule_model.')}
            if hier_sd:
                self.molecule_model.load_state_dict(hier_sd)
                print("   - Hierarchical Backbone: Success")
    
        else:
            print(" > Integrated state_dict not found. Checking individual keys...")
            if 'molecule_model' in checkpoint:
                self.molecule_model.load_state_dict(checkpoint['molecule_model'])
                print("   - Hierarchical Backbone: Success")

    def forward(self, data):
        node_rep = self.molecule_model(data, global_init=None)
        
        atom_mask = (data.node_type == 0)
        frag_mask = (data.node_type == 1)
        global_mask = (data.node_type == 2)
        
        h_atom = node_rep[atom_mask]
        h_atom_pooled = scatter_mean(h_atom, data.batch[atom_mask], dim=0)
        
        h_frag = node_rep[frag_mask]
        h_frag_pooled, _ = scatter_max(h_frag, data.batch[frag_mask], dim=0)
        
        h_global = node_rep[global_mask]
        gnn_combined = torch.cat([h_atom_pooled, h_frag_pooled, h_global], dim=-1)
        
        gnn_emb = self.gnn_proj(gnn_combined)
        fp_emb = self.fp_proj(data.fp_embeddings)
        
        combined = torch.cat([gnn_emb, fp_emb], dim=-1)

        return self.prediction_layer(combined)