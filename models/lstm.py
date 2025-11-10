"""
========================================================================
Model Definition File Contract
========================================================================

This file defines a classifier model architecture that is compatible with
the project's main training (`main_train.py`) and evaluation (`evaluate.py`)
scripts.

To add a new model (e.g., "MyNewModel"), create a new file like this one
(e.g., `models\my_new_model.py`) and implement the following components:

1.  A class that inherits from `torch.nn.Module`.
2.  An `__init__` method with a specific signature.
3.  An *internal* `forward` method for the model's logic.
4.  A `get_scores_for_batch` "adapter" method for training.
5.  A `get_final_scores` "adapter" method for evaluation.

The factory in `models\__init__.py` must also be updated to import
and select this new class based on the `--model_type` argument.

------------------------------------------------------------------------
CONTRACT DETAILS
------------------------------------------------------------------------

--- [1. `__init__` Method] ---

The `__init__` method *must* have the following signature:

def __init__(self, args, vocab_size):
    ...

    - `args`: The fully populated `ArgumentParser` namespace. This
      object will contain all command-line arguments, allowing the
      model to pull its own specific hyperparameters (e.g.,
      `args.my_model_hidden_dim`, `args.my_model_num_layers`).
    
    - `vocab_size`: An integer (e.g., from `len(tokenizer)`)
      specifying the total vocabulary size. This is required to
      correctly initialize the `nn.Embedding` layer.

--- [2. `forward` Method] ---

The `forward` method is *internal* to your model. Its signature can
be whatever you need.
    
    - Example: `def forward(self, inputs, lengths):` (for LSTM)
    - Example: `def forward(self, inputs):` (for Mamba/Transformer)

This method will contain the core architectural logic (embeddings,
RNN/Mamba/Transformer layers, output head).

It *must* be causal (unidirectional) and output a tensor of
per-token logits.

    - **Output Shape:** `(batch_size, seq_len)`

--- [3. `get_scores_for_batch` Method] ---

This is the adapter method called by `main_train.py`. It is
responsible for unpacking the batch, calling its own `forward`
method, and returning *all* per-token scores for the loss
calculation.

    - **Input:** `batch` (The raw, collated batch from the DataLoader.
      Typically `[inputs, lengths, classification_targets]`)
    
    - **Returns:** A tuple of `(scores, targets)`
        - `scores`: `torch.Tensor` of shape `(batch_size, seq_len)`
          (The per-token logits from the `forward` pass).
        - `targets`: `torch.Tensor` of shape `(batch_size,)`
          (The true class labels, e.g., [0, 1, 1, 0]).

--- [4. `get_final_scores` Method] ---

This is the adapter method called by `evaluate.py`. It is
responsible for unpacking the batch, calling `forward`, and
returning the logit from *only* the single, final, unpadded token.

    - **Input:** `batch` (The raw, collated batch, same as above).
    
    - **Returns:** `last_logits`
        - `last_logits`: `torch.Tensor` of shape `(batch_size,)`
          (The logit from the last *real* token for each item
          in the batch).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# --- Model Architecture ---
class LSTMClassifier(nn.Module):
    
    def __init__(self, args, vocab_size):
        """
        Initializes the LSTM model.
        
        Args:
            args: The full ArgumentParser namespace. Reads
                  `args.lstm_hidden_dim` and `args.lstm_num_layers`.
            vocab_size: The total vocabulary size for the embedding layer.
        """
        super().__init__()
        
        # --- CRITICAL CHANGE ---
        # Using `vocab_size` (e.g., 32000) is robust and correct.
        # Using `tokenizer_pad_id + 1` was brittle and would fail
        # with many tokenizers where the pad ID is not the highest ID.
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=args.lstm_hidden_dim, 
            padding_idx=0  # Assuming 0 is your pad ID
        )
        # --- End of Change ---
        
        self.rnn = nn.LSTM(
            args.lstm_hidden_dim, 
            args.lstm_hidden_dim, 
            num_layers=args.lstm_num_layers, 
            bidirectional=False, 
            dropout=0.5,
            batch_first=True # Makes the permute/transpose logic simpler
        )
        self.out_linear = nn.Linear(args.lstm_hidden_dim, 1)

    def forward(self, inputs, lengths):
        """
        Internal forward pass for the LSTM.
        Requires `lengths` for sequence packing.
        """
        # (batch_size, seq_len, hidden_dim)
        embedded_inputs = self.embed(inputs)
        
        # Pack sequence for efficient RNN processing
        packed_inputs = pack_padded_sequence(
            embedded_inputs,
            lengths.cpu(), # Must be on CPU
            batch_first=True,
            enforce_sorted=False
        )
        
        # rnn_output is (packed_batch, hidden_dim)
        rnn_output, _ = self.rnn(packed_inputs)
        
        # Unpack: (batch_size, seq_len, hidden_dim)
        rnn_output, _ = pad_packed_sequence(
            rnn_output, 
            batch_first=True
        )
        
        # (batch_size, seq_len)
        return self.out_linear(rnn_output).squeeze(2)

    # --- 
    # --- Adapter Methods (The "Contract") ---
    # ---
    
    def get_scores_for_batch(self, batch):
        """
        Adapter for training.
        Unpacks batch, calls `self.forward`, and returns all scores.
        """
        # Unpack the batch as needed *by this model*
        inputs, lengths, classification_targets = batch
        
        # Move tensors to the model's device
        inputs = inputs.to(self.embed.weight.device)
        lengths = lengths.to(self.embed.weight.device)
        
        # Call this model's specific forward pass
        scores = self.forward(inputs, lengths)
        
        # Return what the training loop needs
        return scores, classification_targets

    def get_final_scores(self, batch):
        """
        Adapter for evaluation.
        Unpacks batch, calls `self.forward`, and returns final logit.
        """
        # We need all 3 components from the batch
        inputs, lengths, _ = batch 
        
        # Move tensors to the model's device
        inputs = inputs.to(self.embed.weight.device)
        lengths = lengths.to(self.embed.weight.device)
        
        # Call this model's specific forward pass
        # scores shape: (batch_size, seq_len)
        scores = self.forward(inputs, lengths) 
        
        # Find the index of the last token
        # Shape: (batch_size,)
        last_indices = (lengths - 1).long()
        
        # Gather the specific scores from those last indices
        # Shape: (batch_size, 1) -> (batch_size,)
        last_logits = scores.gather(
            1, last_indices.unsqueeze(1)
        ).squeeze(1)
        
        return last_logits