import os
import json
import numpy as np
import torch
import torch.optim as optim
from utils.config_pretrain import args
from models import HierarchicalPretrain, HierarchicalGNN
from torch_geometric.data import Batch, DataLoader

from tqdm import tqdm
from datasets import PretrainDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def save_model(args, model, save_best, step=None):
    if args.output_model_dir == '':
        return

    if save_best:
        suffix = '_best.pth'
    elif step is not None:
        suffix = f'_step{step}.pth'
    else:
        suffix = '_final.pth'
        
    save_path = args.output_model_dir + suffix
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    saver_dict = {
        'model_state_dict': model.state_dict(), 
        'molecule_model': model.molecule_model.state_dict(),
        'step': step 
    }
    
    torch.save(saver_dict, save_path)
    print(f'### Model saved to {save_path} ###')

def train_one_block(args, model, device, loader, optimizer, scheduler, epoch, block_idx, writer, global_step):
    model.train()
    epoch_loss = 0
    block_loss_sum = {
        'total': 0, 'align': 0, 'atom': 0, 
        'frag': 0, 'dist': 0, 'bond': 0
    }
    
    interval_loss = {
        'total': 0, 'align': 0, 'atom': 0, 'frag': 0, 'dist': 0, 'bond': 0
    }
    log_interval = 1000
    
    pbar = tqdm(loader, desc=f"Ep {epoch} | Block {block_idx}")
    for step, batch in enumerate(pbar):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        
        optimizer.zero_grad()
        
        total_loss, l_align, l_atom, l_frag, l_dist, l_bond = model(batch1, batch2)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        with torch.no_grad():
            model.log_vars.clamp_(min=-5.0, max=10.0)
        
        if scheduler is not None:
            scheduler.step()
        
        global_step += 1
        
        for key, val in zip(
            ['total', 'align', 'atom', 'frag', 'dist', 'bond'],
            [total_loss, l_align, l_atom, l_frag, l_dist, l_bond]
        ):
            interval_loss[key] += val.item()
            block_loss_sum[key] += val.item()   
        
        if global_step % 10000 == 0:
            save_model(args, model, save_best=False, step=global_step)
        
        if global_step % log_interval == 0:
            for key in interval_loss:
                writer.add_scalar(f'Loss/Step_{key.capitalize()}', interval_loss[key] / log_interval, global_step)
            
            writer.add_scalar('Stat/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            with torch.no_grad():
                w = torch.exp(-model.log_vars).cpu()
                writer.add_scalar('Weight/Align', w[0], global_step)
                writer.add_scalar('Weight/Atom', w[1], global_step)
                writer.add_scalar('Weight/Frag', w[2], global_step)
                writer.add_scalar('Weight/Dist', w[3], global_step)
                writer.add_scalar('Weight/Bond', w[4], global_step)
            
            for key in interval_loss: interval_loss[key] = 0
            
            pbar.set_postfix({
                'Step': global_step,
                'L': f'{total_loss.item():.3f}',
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
        epoch_loss += total_loss.item()
    
    n_steps = len(loader)
    avg_block_loss = {k: v / n_steps for k, v in block_loss_sum.items()}
    return epoch_loss / n_steps, global_step, avg_block_loss

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    pcq_data_dir = "../datasets/pcqm4m-v2_dense/"
    vocab_path = os.path.join(pcq_data_dir, "pcqm_vocab.json")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"Vocab Loaded: {vocab_size} tokens")
    print(f"Args.type = {args.gnn_type}")
    
    molecule_model = HierarchicalGNN(args.num_layer, args.emb_dim, vocab_size=vocab_size, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    model = HierarchicalPretrain(args, molecule_model, vocab_size=vocab_size).to(device)
    
    args.lr = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr,             # Peak learning rate
        total_steps=329000,         # Total number of training steps
        pct_start=0.1,              # Fraction of steps for LR warmup (top 10%)
        anneal_strategy='cos',      # Cosine annealing after peak
        div_factor=10,              # Initial LR = max_lr / div_factor (i.e., starts at 1e-5)
        final_div_factor=100        # Final LR = max_lr / (div_factor * final_div_factor)
    ) 
    
    writer = SummaryWriter(os.path.join('runs', f'PCQMv2_Pretrain_{datetime.now().strftime("%m%d_%H%M")}'))

    global_step = 0
    optimal_loss = 1e10
    total_blocks = 34 # 0~33
    
    print(f"\nTraining Configuration:")
    print(f" - Learning Rate: {args.lr}")
    print(f" - Weight Decay: {args.decay}")
    print(f" - Batch Size: {args.batch_size}")
    print(f" - Epochs: {args.epochs}")
    
    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        epoch_loss = 0
        
        epoch_loss_sum = {
            'total': 0, 'align': 0, 'atom': 0,
            'frag': 0, 'dist': 0, 'bond': 0
        }
        
        # Block Loop
        block_indices = list(range(total_blocks))
        np.random.shuffle(block_indices)    
        
        for b_idx in block_indices:
            block_path = os.path.join(pcq_data_dir, f'pcq_block_{b_idx}.pt')
            print(f"Loading {block_path}...")
            
            # Block Loading
            data_list = torch.load(block_path)
            dataset = PretrainDataset(
                data_list, 
                None,
                mask_ratio_atom=0.25, 
                mask_ratio_frag=0.15, 
                mask_token=vocab_size
            )
            
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            
            # Pretraining
            b_loss, global_step, avg_block_loss = train_one_block(args, model, device, loader, optimizer, scheduler, epoch, b_idx, writer, global_step)
            epoch_loss += b_loss
            
            for key in epoch_loss_sum:
                epoch_loss_sum[key] += avg_block_loss[key]
            
            # Clean up memory
            del data_list, dataset, loader
            torch.cuda.empty_cache()
            
            if global_step >= 1000000:
                print("Reached 1,000,000 steps. Finishing training.")
                break
        
        if global_step >= 1000000: 
            break
        
        avg_epoch_loss = epoch_loss / total_blocks
        writer.add_scalar('Loss/Epoch_Avg', avg_epoch_loss, epoch)
        
        for key in ['align', 'atom', 'frag', 'dist', 'bond']:
            writer.add_scalar(
                f'Loss/Epoch_{key.capitalize()}',
                epoch_loss_sum[key] / total_blocks,
                epoch
            )
        
        print(f"\n[Epoch {epoch} Summary]")
        print(f"  Total : {avg_epoch_loss:.4f}")
        for key in ['align', 'atom', 'frag', 'dist', 'bond']:
            print(f"  {key.capitalize():<6}: {epoch_loss_sum[key] / total_blocks:.4f}")

        if avg_epoch_loss < optimal_loss:
            optimal_loss = avg_epoch_loss
            save_model(args, model, save_best=True, step=global_step)
        
    writer.close()
    save_model(args, model, save_best=False)
    