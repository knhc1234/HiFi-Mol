import argparse

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)

# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='GEOM_2D_nmol50000_nconf5_nupper1000')
parser.add_argument('--num_workers', type=int, default=8)

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)     
parser.add_argument('--decay', type=float, default=0.0)

# about HierarchicalGNN
parser.add_argument('--gnn_type', type=str, default='gin')
parser.add_argument('--num_layer', type=int, default=5)         
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.1) 
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')

parser.add_argument('--output_model_dir', type=str, default='./Pretrain/HiFi-Mol_GIN')

# verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

args = parser.parse_args()
print('arguments\t', args)
