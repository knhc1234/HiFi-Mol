from os.path import join
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from utils.config import args
from utils.splitters import random_scaffold_split, random_split, scaffold_split
from utils.util import get_num_task
from models import HierarchicalGNN, HiFi_Mol
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.data import DataLoader, Data
from datasets import MoleculeDataset, HiFi_Mol_Downstream_Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
import itertools
import csv
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
torch.multiprocessing.set_sharing_strategy('file_system')

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        if args.dataset in ['qm7', 'qm8', 'qm9']:
            loss = torch.sum(torch.abs(pred-y))/y.size(0)
        elif args.dataset in ['esol','freesolv','lipophilicity']:
            loss = torch.sum((pred-y)**2)/y.size(0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().item()
        
    print(f"Total Loss: {total_loss:.4f}")
    return total_loss

def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            
        true = batch.y.view(pred.shape)
        y_true.append(true)
        y_scores.append(pred)
    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse=np.sqrt(mse)
    return mse, mae, rmse

def run_experiment(title, lr, batch_size, epochs, decay, seed):
    args.runseed = seed
    args.lr = lr
    args.batch_size = batch_size
    args.epochs = epochs
    args.decay = decay
    
    print(f"\n>>> Running Experiment: Seed={seed}, LR={lr}, Batch={batch_size}, Epochs={epochs}, Decay={decay}")

    title2 = f'{title}_{args.dataset}_seed{args.runseed}'

    log_dir = os.path.join('runs', title2)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print("GPU usable")
        torch.cuda.manual_seed_all(args.runseed)
    else:
        print("GPU unusable")

    num_tasks = get_num_task(args.dataset)
    dataset_folder = args.input_data_dir    #'../datasets/molecule_datasets/'
    dataset_raw = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
    fp_embedding_mmap = np.load(dataset_folder + args.dataset + '/processed/fp_embeddings.npy', mmap_mode='r')
    dataset = HiFi_Mol_Downstream_Dataset(dataset_raw, fp_embedding_mmap)
    
    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,
                                                                    frac_valid=0.1, frac_test=0.1)
        print('Split by Scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
                                                                frac_test=0.1, seed=args.seed)
        print('Split by Random')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
                                                                        frac_train=0.8, frac_valid=0.1,
                                                                        frac_test=0.1, seed=args.seed)
    else:
        raise ValueError('Unvalid Option')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    molecule_model = HierarchicalGNN(args.num_layer, args.emb_dim, vocab_size=18907, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    model = HiFi_Mol(args=args, num_tasks=num_tasks, molecule_model=molecule_model)
    #pretrained_path = f"./Pretrain/HiFi-Mol_GIN_final.pth"
    model.from_pretrained(args.input_model_file, device)
    model.to(device)

    backbone_params = []
    new_layer_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
        if name.startswith("molecule_model."):
            backbone_params.append(param)
        else:
            new_layer_params.append(param)
    
    optimizer = optim.AdamW(
        [{'params': backbone_params, 'lr': args.lr * args.lr_scale},  
        {'params': new_layer_params, 'lr': args.lr}       
        ], 
        lr=args.lr,
        weight_decay=args.decay, 
        betas=(0.9, 0.999))
    
    print("total parameters:", sum(p.numel() for p in model.parameters()))
    print("learnable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    best_val_rmse = 1e10
    final_test_rmse = 1e10

    for epoch in range(1, args.epochs + 1):
        loss = train(model, device, train_loader, optimizer)
        writer.add_scalar('Loss/train_total', loss, epoch)
        
        train_mse, train_mae, train_rmse = eval(model, device, train_loader)
        writer.add_scalar('RMSE/train', train_rmse, epoch)
        
        val_mse, val_mae, val_rmse = eval(model, device, val_loader)
        writer.add_scalar('RMSE/validation', val_rmse, epoch)
        
        test_mse, test_mae, test_rmse = eval(model, device, test_loader)
        writer.add_scalar('RMSE/test', test_rmse, epoch)
        
        print('Epoch {:d}\t train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(epoch, train_rmse, val_rmse, test_rmse))
        print()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            final_test_rmse = test_rmse 
            
            if not args.output_model_dir == '':
                if not os.path.exists(args.output_model_dir):
                    os.makedirs(args.output_model_dir, exist_ok=True)
                output_model_path = join(args.output_model_dir, f'{title2}.pth')
                saved_model_dict = {
                    'molecule_model': molecule_model.state_dict(),
                    'model': model.state_dict()
                }
                torch.save(saved_model_dict, output_model_path)
            
    writer.close()
    print(f"Test RMSE Result: {final_test_rmse}")

    return {
        "Seed": seed,
        "LR": lr,
        "BatchSize": batch_size,
        "Epochs": epochs,
        "WeightDecay": decay,
        "Best_Val_RMSE": best_val_rmse,
        "Test_RMSE_at_Best_Val": final_test_rmse
    }

import gc

if __name__ == '__main__':
    DATASET_CONFIGS = {
        'freesolv': {
            'lrs': [1e-4],
            'batch_sizes': [32],
            'epochs_list': [1],
            'decays': [0]
        },
        'esol': {
            'lrs': [1e-4],
            'batch_sizes': [32],
            'epochs_list': [1],
            'decays': [0]
        },
        'lipophilicity': {
            'lrs': [1e-4],
            'batch_sizes': [32],
            'epochs_list': [1],
            'decays': [0]
        },
    }
    runseeds = [0]
    
    for dataset, cfg in DATASET_CONFIGS.items():
        args.dataset = dataset
        title = f"HiFi-Mol"
        csv_file = f"{title}_{args.dataset}.csv"

        combinations = list(itertools.product(
            cfg['lrs'], 
            cfg['batch_sizes'], 
            cfg['epochs_list'], 
            cfg['decays'], 
            runseeds
        ))
        
        # CSV 헤더 작성
        with open(csv_file, mode='w', newline='') as f:
            writer_main = csv.DictWriter(f, fieldnames=["Seed", "LR", "BatchSize", "Epochs", "WeightDecay", "Best_Val_RMSE", "Test_RMSE_at_Best_Val"])
            writer_main.writeheader()

            for lr, bs, ep, dec, seed in combinations:
                result = run_experiment(title, lr, bs, ep, dec, seed)
                writer_main.writerow(result)
                f.flush() 
                
                torch.cuda.empty_cache()
                gc.collect()