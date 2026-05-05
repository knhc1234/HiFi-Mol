import argparse

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=1)

# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='../datasets/molecule_datasets/')
parser.add_argument('--dataset', type=str, default='bace')
parser.add_argument('--num_workers', type=int, default=8)

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)     
parser.add_argument('--lr_scale', type=float, default=0.1)
parser.add_argument('--decay', type=float, default=0)

# about HierarchicalGNN
parser.add_argument('--gnn_type', type=str, default='gin')
parser.add_argument('--num_layer', type=int, default=5)         
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.1) 
parser.add_argument('--graph_pooling', type=str, default='mean') 
parser.add_argument('--JK', type=str, default='last')

parser.add_argument('--input_model_file', type=str, default='./Pretrain/HiFi-Mol_GIN_final.pth')
parser.add_argument('--output_model_dir', type=str, default='./results/HiFi-Mol_last')
args = parser.parse_args()
print('arguments\t', args)
