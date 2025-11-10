import os
import random
import time
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from data import Dataset
from models import get_model # <-- Import the factory
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from constants import *
from huggingface_hub import login

def train(model, dataset, optimizer, criterion, epoch, args):
    model.train()
    dataset.shuffle('train', seed=epoch + args.seed)
    
    loader = dataset.loader('train', args.batch_size, num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Training: ')

    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        # Unpack the simplified batch
        inputs, lengths, classification_targets = batch
        
        # Move to device
        scores, classification_targets = model.get_scores_for_batch(batch)
        classification_targets = classification_targets.to(args.device)

        # Get per-token scores from the model
        scores = model(inputs, lengths) # (batch_size, seq_len)

        # --- This is the "Implicit Prefix" Loss Logic ---
        
        expanded_labels = classification_targets.unsqueeze(1).expand(-1, scores.shape[1])
        
        # 2. Get padding mask (batch_size, seq_len)
        length_mask = pad_mask(lengths) # 1 for real tokens, 0 for padding
        
        # 3. Flatten scores, labels, and mask
        scores_flat = scores.flatten()
        labels_flat = expanded_labels.flatten().float()
        mask_flat = length_mask.flatten()
        
        # 4. Select only the non-padded tokens for loss calculation
        scores_unpadded = scores_flat[mask_flat == 1]
        labels_unpadded = labels_flat[mask_flat == 1]
        
        # 5. Calculate loss
        loss = criterion(scores_unpadded, labels_unpadded)
        # --- End of Implicit Logic ---

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), inputs.size(0))
        if batch_num % args.print_freq == 0:
            progress.display(batch_num)
            
    progress.display(total_length)

def validate(model, dataset, criterion, args):
    model.eval()
    loader = dataset.loader('val', args.batch_size, num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Validation: ')

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
            inputs, lengths, classification_targets = batch
            inputs = inputs.to(args.device)
            lengths = lengths.to(args.device)
            classification_targets = classification_targets.to(args.device)

            scores = model(inputs, lengths)

            # --- Identical Implicit Loss Logic ---
            expanded_labels = classification_targets.unsqueeze(1).expand(-1, scores.shape[1])
            length_mask = pad_mask(lengths)
            
            scores_flat = scores.flatten()
            labels_flat = expanded_labels.flatten().float()
            mask_flat = length_mask.flatten()
            
            scores_unpadded = scores_flat[mask_flat == 1]
            labels_unpadded = labels_flat[mask_flat == 1]
            
            if scores_unpadded.nelement() > 0: # Avoid empty batches if all are filtered
                loss = criterion(scores_unpadded, labels_unpadded)
                loss_meter.update(loss.item(), inputs.size(0))

    progress.display(total_length)
    return loss_meter.avg


def main(args):
    login(token=args.hf_token)

    # Hard-code the task
    args.task = 'transfer'
    args.device = torch.device(DEVICE)

    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)
    
    model = get_model(args, dataset.tokenizer_pad_id) # <-- Factory call!
    model = model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_metric = 1e8 # Lower is better for BCE
    
    print('Model Parameters:', num_params(model))
    criterion = nn.BCEWithLogitsLoss().to(args.device)
    
    now = datetime.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(args.epochs):
        print(f"--- TRAINING: Epoch {epoch} at {time.ctime()} ---")
        train(model, dataset, optimizer, criterion, epoch, args)
        
        print(f"--- VALIDATION: Epoch {epoch} at {time.ctime()} ---")
        metric = validate(model, dataset, criterion, args)

        if metric < best_val_metric:
            print(f'New best val metric: {metric:.4f}')
            best_val_metric = metric
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_metric': best_val_metric,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(args.save_dir, f'{args.model_type}_{date_string}.pth.tar'))

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # DATA
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="Dir with train/holmes, train/encyclopedia, etc.")
    
    # SAVE/LOAD
    parser.add_argument('--save_dir', type=str, required=True, help='where to save ckpts')
    parser.add_argument('--hf_token', type=str, required=True, help='HuggingFace Token')
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')

    # TRAINING
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4) # 1e-3 was a bit high
    parser.add_argument('--seed', type=int, default=24601, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers for data loader')
    parser.add_argument('--pos_cat', type=str, default='holmes', help='Name of positive category for training - must match filename in data_dir')
    parser.add_argument('--neg_cat', type=str, default='encyclopedia', help='Name of negative category for training - must match filename in data_dir')

    # PRINTING
    parser.add_argument('--print_freq', type=int, default=100, 
                        help='how often to print metrics (every X batches)')
    
    # MODEL
    parser.add_argument(
        '--model_type', 
        type=str, 
        default='lstm', 
        choices=['lstm']
    )
    
    # --- LSTM-Specific Hyperparameters ---
    parser.add_argument('--lstm_hidden_dim', type=int, default=300)
    parser.add_argument('--lstm_num_layers', type=int, default=3)

    # --- Mamba-Specific Hyperparameters ---
    parser.add_argument('--mamba_d_model', type=int, default=256)
    parser.add_argument('--mamba_n_layer', type=int, default=4)
    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_expand', type=int, default=2)
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)