import os
import random
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from huggingface_hub import login
from transformers import AutoTokenizer
from collections import defaultdict
from types import SimpleNamespace
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import requests
from pathlib import Path
from argparse import ArgumentParser

import os
import torch
import random
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from data import Dataset
from models import get_model # <-- Import the factory
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from constants import *


def train(model, dataset, optimizer, criterion, epoch, args):
    model.train()
    dataset.shuffle('train', seed=epoch + args.seed)

    loader = dataset.loader('train', args.batch_size, num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Training: ')

    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        # 1. Use the Contract to get scores and targets
        # (The model handles device movement and internal unpacking)
        scores, classification_targets = model.get_scores_for_batch(batch)

        # 2. We still need 'lengths' for the mask logic below
        _, lengths, _ = batch
        lengths = lengths.to(scores.device) # Ensure it matches the GPU/CPU of scores

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

        # --- SANITY CHECK ---
        if dataset.tokenizer_pad_id is not None:
            # Assuming the model has an 'embed' layer.
            # If you change model types (e.g. to Mamba), ensure this attribute path exists.
            pad_grad = model.embed.weight.grad[dataset.tokenizer_pad_id]

            # It's possible pad_grad is None if the batch had NO padding (unlikely but possible)
            if pad_grad is not None:
                if torch.sum(torch.abs(pad_grad)) > 0:
                    raise RuntimeError(f"FATAL: Model is learning from PAD token {dataset.tokenizer_pad_id}!")
        # --------------------

        optimizer.step()

        loss_meter.update(loss.item(), scores.size(0))
        if batch_num % args.print_freq == 0:
            progress.display(batch_num)

    progress.display(total_length)
    
def validate(model, dataset, criterion, args):
    model.eval()
    loader = dataset.loader('val', args.batch_size, num_workers=args.num_workers)

    loss_meter = AverageMeter('loss', ':6.4f')
    acc_meter = AverageMeter('acc', ':6.2f') # Add an accuracy meter

    total_length = len(loader)
    # Add acc_meter to the display list
    progress = ProgressMeter(total_length, [loss_meter, acc_meter], prefix='Validation: ')

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):

            # 1. Use the Contract (Handles device movement & unpacking)
            scores, classification_targets = model.get_scores_for_batch(batch)

            # 2. Get lengths specifically for masking (Manual step)
            _, lengths, _ = batch
            lengths = lengths.to(scores.device)

            # --- Identical Implicit Loss Logic ---
            expanded_labels = classification_targets.unsqueeze(1).expand(-1, scores.shape[1])
            length_mask = pad_mask(lengths)

            scores_flat = scores.flatten()
            labels_flat = expanded_labels.flatten().float()
            mask_flat = length_mask.flatten()

            scores_unpadded = scores_flat[mask_flat == 1]
            labels_unpadded = labels_flat[mask_flat == 1]

            if scores_unpadded.nelement() > 0:
                # Calculate Loss
                loss = criterion(scores_unpadded, labels_unpadded)

                # --- Calculate Accuracy ---
                # Sigmoid converts logits to probs (0 to 1)
                probs = torch.sigmoid(scores_unpadded)
                # Anything > 0.5 is a prediction for Class 1 (EB11)
                preds = (probs > 0.5).float()
                # Compare preds to true labels
                acc = (preds == labels_unpadded).float().mean()
                # --------------------------

                # Update meters (Use scores.size(0) for batch size)
                loss_meter.update(loss.item(), scores.size(0))
                acc_meter.update(acc.item(), scores.size(0))

    progress.display(total_length)

    # Return loss (or accuracy, depending on what you want to track for early stopping)
    return loss_meter.avg

def dry_run(args, model, dataset):
    print("--- INITIATING DRY RUN ---")

    # Ensure model is in training mode so gradients are generated
    model.train()

    # 1. Get one batch
    loader = dataset.loader('train', batch_size=2)
    batch = next(iter(loader))

    # 2. Forward Pass
    # Your model's 'get_scores_for_batch' handles moving tensors to GPU,
    # so we don't need to manually .to(device) the batch here.
    scores, targets = model.get_scores_for_batch(batch)

    # 3. Backward Pass (Dummy Loss)
    # We just need a scalar to test that backprop works without OOMing.
    # We don't need the real loss function here.
    loss = scores.sum()
    loss.backward()

    # 4. Check Max ID vs Vocab Size (The critical check for your bug)
    input_ids = batch[0]
    # Access the embedding weight size directly
    vocab_limit = model.embed.weight.size(0)
    if input_ids.max() >= vocab_limit:
        raise ValueError(f"CRITICAL: Batch contains Token ID {input_ids.max()}, but Embedding size is only {vocab_limit}. Adjust get_model()!")

    # 5. Save/Load Check (Verifies serialization)
    torch.save(model.state_dict(), "dry_run.pth")
    model.load_state_dict(torch.load("dry_run.pth"))

    print("--- DRY RUN PASSED ---")

    # Clean up
    if os.path.exists("dry_run.pth"):
        os.remove("dry_run.pth")

    # Zero gradients so this test doesn't affect the first real training step
    model.zero_grad()
    
def evaluate_test_set(model, dataset, args, checkpoint_path):
    print(f"\n--- STARTING FINAL EVALUATION ---")
    print(f"Loading best checkpoint from: {checkpoint_path}")

    # Load the best weights
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    loader = dataset.loader('test', args.batch_size, num_workers=args.num_workers)

    all_preds = []
    all_targets = []

    print("Running inference on Test Set...")
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            # 1. Get the logits for the LAST token in the sequence
            # We use get_final_scores because we care about the document-level decision,
            # not the 'loss at every step' intermediate predictions.
            logits = model.get_final_scores(batch)

            # 2. Convert logits to probabilities and then to binary predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            # 3. Get actual labels (Index 2 in your collate tuple)
            # (inputs, lengths, classification_labels)
            targets = batch[2]

            # 4. Move to CPU and collect
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # --- Calculate Metrics ---
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    conf_mat = confusion_matrix(all_targets, all_preds)

    print("\n" + "="*30)
    print("TEST SET RESULTS")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(conf_mat)
    print("-" * 30)

    # Detailed report useful for seeing class imbalances
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Negative', 'Positive']))

    return f1

def seed_everything(seed=42):
    # 1. Set the python built-in random seed
    random.seed(seed)

    # 2. Set the numpy seed
    np.random.seed(seed)

    # 3. Set the pytorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # If using multi-GPU

    # 4. Important: Force CuDNN to be deterministic
    # This slows down training slightly but ensures 'exact' reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. Set hashing seed (vital for dictionary ordering/hashing)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Global seed set to {seed}")

def main():

    seed_everything(SEED)

    args = setup()
    login(token=args.hf_token)

    # Hard-code the task
    args.task = 'transfer'
    args.device = torch.device(DEVICE)

    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    model = get_model(args, len(dataset.tokenizer), dataset.tokenizer_pad_id)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- ADDED: Scheduler ---
    # If validation loss doesn't improve for 2 epochs, cut LR by half
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_metric = 1e8

    print('Model Parameters:', num_params(model))
    criterion = nn.BCEWithLogitsLoss().to(args.device)

    dry_run(args, model, dataset)

    now = datetime.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(args.save_dir, f'{args.model_type}_{date_string}.pth.tar')

    # --- ADDED: Early Stopping Config ---
    early_stopping_patience = 5
    epochs_no_improve = 0
    # ------------------------------------

    for epoch in range(args.epochs):
        print(f"--- TRAINING: Epoch {epoch} at {time.ctime()} ---")
        train(model, dataset, optimizer, criterion, epoch, args)

        print(f"--- VALIDATION: Epoch {epoch} at {time.ctime()} ---")
        metric = validate(model, dataset, criterion, args)

        # Update the scheduler based on validation loss
        scheduler.step(metric)

        # --- ADD THIS TO REPLACE VERBOSE=TRUE ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        # ----------------------------------------

        if metric < best_val_metric:
            print(f'New best val metric: {metric:.4f}')
            best_val_metric = metric

            # Reset patience because we improved
            epochs_no_improve = 0

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_metric': best_val_metric,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, save_path)
        else:
            # We didn't improve
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after epoch {epoch}!")
                print(f'Final val metric: {best_val_metric:.4f}')
                break
    evaluate_test_set(model, dataset, args, save_path)

if __name__ == '__main__':
    
    SEED=24601

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_TYPE="mamba"

    DATA_DIR="data"
    SAVE_DIR="trained_models"

    HF_TOKEN="XXXXX" # Set automatically from secret if on colab

    BATCH_SIZE=64
    EPOCHS=100
    LR=0.0001
    NUM_WORKERS=4
    CKPT=None
    PRINT_FREQ=100

    # --- Tokenizer ---
    TOKENIZER_NAME = 'meta-llama/Llama-3.2-3B-Instruct'

    # --- Data Processing ---
    POS_CAT="eb"
    NEG_CAT="simple_wiki"
    VAL_SIZE = 400 # total across both datasets
    MAX_LEN = 1024
    MIN_SENTENCE_LENGTH = 3

    # --- LSTM Specific ---
    LSTM_HIDDEN_DIM=128
    LSTM_NUM_LAYERS=4

    # --- Mamba Specific ---
    MAMBA_D_MODEL=128
    MAMBA_NUM_LAYERS=4
    MAMBA_D_STATE=16
    MAMBA_DROPOUT=0.1


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
    

    training_args = SimpleNamespace(
        data_dir=DATA_DIR,
        save_dir=SAVE_DIR,
        hf_token=HF_TOKEN,
        ckpt=CKPT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        seed=SEED,
        num_workers=NUM_WORKERS,
        pos_cat=POS_CAT,
        neg_cat=NEG_CAT,
        print_freq=PRINT_FREQ,
        model_type=MODEL_TYPE,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_num_layers=LSTM_NUM_LAYERS,
        mamba_d_model=MAMBA_D_MODEL,
        mamba_num_layers=MAMBA_NUM_LAYERS,
        mamba_d_state=MAMBA_D_STATE,
        mamba_dropout=MAMBA_DROPOUT,
        on_colab=False,
        val_size=VAL_SIZE,
        max_len=MAX_LEN,
        min_sentence_length=MIN_SENTENCE_LENGTH,
        tokenizer_name=TOKENIZER_NAME
    )

    if 'google.colab' in sys.modules:
        training_args.on_colab = True
        from google.colab import drive, userdata
        drive.mount('/content/drive')
        training_args.hf_token = userdata.get('HF_TOKEN')
        training_args.save_dir = "/content/drive/My Drive/"


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)