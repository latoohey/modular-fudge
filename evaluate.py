import os
import random
import time
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from data import Dataset
from models import get_model  # <-- IMPORT THE FACTORY
from util import num_params
from constants import *
from huggingface_hub import login

def evaluate(args):
    login(token=args.hf_token)
    # --- 1. Load Dataset ---
    print("Loading test data...")
    dataset = Dataset(args) # This needs args.data_dir, args.hf_token
    loader = dataset.loader(
        'test', 
        args.batch_size, 
        num_workers=args.num_workers
    )
    
    # --- 2. Load Checkpoint & Model ---
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    
    # Get the args *saved in the checkpoint*
    # This has all the model info (model_type, hidden_dim, etc.)
    model_args = checkpoint['args']
    
    # Re-initialize the model using the factory
    # The factory will read model_args.model_type and build the correct model
    model = get_model(model_args, dataset.tokenizer_pad_id) # <-- USE THE FACTORY
    
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)
    model.eval() # Set model to evaluation mode
    
    print('Model Parameters:', num_params(model))
    print(f"Checkpoint loaded (trained for {checkpoint['epoch']} epochs).")

    all_preds = []
    all_labels = []

    # --- 3. Run Evaluation Loop ---
    with torch.no_grad(): # Disable gradient calculation
        for batch in tqdm(loader, desc="Evaluating"):
            # Move the whole batch to the device
            batch = [t.to(args.device) for t in batch]
            
            # Get the true labels from the batch
            # (Assuming batch is [inputs, lengths, classification_targets])
            classification_targets = batch[2]

            # --- This is the key abstraction ---
            # Call the model's adapter to get only the final scores
            # The model itself is responsible for its own forward pass
            # and finding the last token.
            # Shape: (batch_size,)
            last_logits = model.get_final_scores(batch)
            
            # Convert logits to final 0/1 predictions
            preds = (torch.sigmoid(last_logits) > 0.5).long()
            
            # Store predictions and true labels
            all_preds.append(preds.cpu())
            all_labels.append(classification_targets.cpu())

    # --- 4. Calculate and Print Metrics ---
    print("\n--- Evaluation Report ---")
    
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    
    # Use sklearn to generate the report
    target_names = ['style_0', 'style_1']
    print(classification_report(
        all_labels_tensor.numpy(), 
        all_preds_tensor.numpy(), 
        target_names=target_names
    ))

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # These args are for the *evaluation script*
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="Dir with test/formal, test/informal, etc.")
    parser.add_argument('--ckpt', type=str, required=True, 
                        help='Path to the model_best.pth.tar file')
    parser.add_argument('--hf_token', type=str, required=True, 
                        help='HuggingFace Token (for tokenizer)')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=24601, help='random seed')
    parser.add_argument('--pos_cat', type=str, default='holmes', help='Name of positive category for training - must match filename in data_dir')
    parser.add_argument('--neg_cat', type=str, default='encyclopedia', help='Name of negative category for training - must match filename in data_dir')
    
    args = parser.parse_args()

    # Set device
    args.device = torch.device(DEVICE)
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    evaluate(args)