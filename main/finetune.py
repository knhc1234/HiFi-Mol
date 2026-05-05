from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils.config import args
from utils.splitters import random_scaffold_split, random_split, scaffold_split
from utils.util import get_num_task
from models import HierarchicalGNN, HiFi_Mol
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader, Data
from datasets import MoleculeDataset, HiFi_Mol_Downstream_Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import itertools
import csv
import torch.multiprocessing
import gc
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
torch.multiprocessing.set_sharing_strategy('file_system')

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_valid_samples = 0
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        is_valid = y ** 2 > 0
        sample_valid_mask = is_valid.any(dim=1)
        num_valid_samples = sample_valid_mask.sum().item()
        
        if num_valid_samples == 0: 
            continue
        
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().item() * num_valid_samples
        total_valid_samples += num_valid_samples
    
    if total_valid_samples > 0:
        avg_loss = total_loss / total_valid_samples
    else:
        avg_loss = 0
    
    print(f"Total Loss: {avg_loss:.4f}")
    return avg_loss

def eval(model, device, loader, criterion):
    model.eval()
    y_true, y_scores = [], []
    total_loss = 0
    sample_count = 0    

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            
            y = batch.y.view(pred.shape).to(torch.float64)
            is_valid = y ** 2 > 0
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            loss_mat = torch.where(
                is_valid, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
            
            total_loss += torch.sum(loss_mat).item()
            sample_count += torch.sum(is_valid).item()

        true = batch.y.view(pred.shape)
        y_true.append(true)
        y_scores.append(pred)
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    avg_roc = sum(roc_list) / len(roc_list) if len(roc_list) > 0 else 0
    avg_loss = total_loss / sample_count if sample_count > 0 else 0
    
    return avg_roc, avg_loss, y_true, y_scores

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
    
    print("Pre-calculating chemical properties...")
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    molecule_model = HierarchicalGNN(args.num_layer, args.emb_dim, vocab_size=18907, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    model = HiFi_Mol(args=args, num_tasks=num_tasks, molecule_model=molecule_model)
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
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    print("total parameters:", sum(p.numel() for p in model.parameters()))
    print("learnable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    best_val_roc = -1
    final_test_roc = 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer, criterion)
        writer.add_scalar('Loss/train_total', loss_acc, epoch)
        
        train_roc, _, _, _ = eval(model, device, train_loader, criterion)
        writer.add_scalar('AUC/train', train_roc, epoch)
        
        val_roc, _, _, _ = eval(model, device, val_loader, criterion)
        writer.add_scalar('AUC/validation', val_roc, epoch)
        
        test_roc, _, _, _ = eval(model, device, test_loader, criterion)
        writer.add_scalar('AUC/test', test_roc, epoch)
        
        print('Epoch {:d}\t train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(epoch, train_roc, val_roc, test_roc))
        print()

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            final_test_roc = test_roc 
            
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
    print(f"Test AUC Result: {final_test_roc}")
    return {
        "Seed": seed,
        "LR": lr,
        "BatchSize": batch_size,
        "Epochs": epochs,
        "WeightDecay": decay,
        "Best_Val_AUC": best_val_roc,
        "Test_AUC_at_Best_Val": final_test_roc
    }

if __name__ == '__main__':
    DATASET_CONFIGS = {
        'bbbp': {
            'lrs': [1e-4],
            'batch_sizes': [128],
            'epochs_list': [20],
            'decays': [0]
        },
        'bace': {
            'lrs': [1e-5],
            'batch_sizes': [32],
            'epochs_list': [20],
            'decays': [0]
        },
        'sider': {
            'lrs': [1e-4],
            'batch_sizes': [32],
            'epochs_list': [20],
            'decays': [0]
        },
        'clintox': {
            'lrs': [1e-5],
            'batch_sizes': [32],
            'epochs_list': [100],
            'decays': [0]
        },
        'tox21': {
            'lrs': [1e-5],
            'batch_sizes': [32],
            'epochs_list': [100],
            'decays': [0]
        },
        'toxcast': {
            'lrs': [1e-4],
            'batch_sizes': [32],
            'epochs_list': [100],
            'decays': [0]
        },
        'hiv': {
            'lrs': [1e-5],
            'batch_sizes': [32],
            'epochs_list': [20],
            'decays': [0]
        },
        'muv': {
            'lrs': [1e-4],
            'batch_sizes': [32],
            'epochs_list': [10],
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
        
        with open(csv_file, mode='w', newline='') as f:
            writer_main = csv.DictWriter(f, fieldnames=["Seed", "LR", "BatchSize", "Epochs", "WeightDecay", "Best_Val_AUC", "Test_AUC_at_Best_Val"])
            writer_main.writeheader()

            for lr, bs, ep, dec, seed in combinations:
                result = run_experiment(title, lr, bs, ep, dec, seed)
                writer_main.writerow(result)
                f.flush() 
                torch.cuda.empty_cache()
                gc.collect()