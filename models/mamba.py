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

def __init__(self, args, vocab_size, pad_id):
    ...

    - `args`: The fully populated `ArgumentParser` namespace. This
      object will contain all command-line arguments, allowing the
      model to pull its own specific hyperparameters (e.g.,
      `args.my_model_hidden_dim`, `args.my_model_num_layers`).

    - `vocab_size`: An integer (e.g., from `len(tokenizer)`)
      specifying the total vocabulary size.

    - `pad_id`: The integer index of the padding token.
      The model MUST use this to set `padding_idx` in the
      `nn.Embedding` layer. This ensures the embedding for
      padding is fixed to zero, which is critical for
      proper loss calculation and gradient safety.

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

    - **Note on Padding:** The model is NOT required to mask/zero-out
      logits at padding positions. The training loop (`main_train.py`)
      applies a strict mask to the loss function based on sequence
      lengths. Therefore, it is acceptable for the model to output
      noise at padding indices (common in unmasked Mamba/RNNs),
      provided the causal history of *valid* tokens remains intact.

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
    *** IMPORTANT ***
    Both `scores` and `targets` MUST be returned on the same
    DEVICE (GPU/CPU) as the model. The training loop assumes
    this adapter handles all device movement.

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

class MambaClassifier(nn.Module):

    def __init__(self, args, vocab_size, pad_id):
        """
        Initializes the Mamba model.

        Args:
            args: The full ArgumentParser namespace. Reads:
                  - `args.mamba_d_model` (hidden dimension, default 256)
                  - `args.mamba_d_state` (SSM state dimension, default 16)
                  - `args.mamba_d_conv` (local convolution width, default 4)
                  - `args.mamba_expand` (expansion factor, default 2)
                  - `args.mamba_num_layers` (number of Mamba blocks, default 4)
                  - `args.mamba_dropout` (dropout rate, default 0.1)
            vocab_size: The total vocabulary size for the embedding layer.
        """
        super().__init__()

        # Get hyperparameters from args with defaults
        self.d_model = getattr(args, 'mamba_d_model', 256)
        self.d_state = getattr(args, 'mamba_d_state', 16)
        self.d_conv = getattr(args, 'mamba_d_conv', 4)
        self.expand = getattr(args, 'mamba_expand', 2)
        self.num_layers = getattr(args, 'mamba_num_layers', 4)
        self.dropout_rate = getattr(args, 'mamba_dropout', 0.1)

        # Embedding layer
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.d_model,
            padding_idx=pad_id  # Use the REAL pad ID
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)

        # Stack of Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=self.d_model,    # Model dimension
                d_state=self.d_state,    # SSM state expansion factor
                d_conv=self.d_conv,      # Local convolution width
                expand=self.expand,      # Block expansion factor
            )
            for _ in range(self.num_layers)
        ])

        # Layer normalization between blocks
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.num_layers)
        ])

        # Final layer norm before output
        self.final_norm = nn.LayerNorm(self.d_model)

        # Output projection to single logit per token
        self.out_linear = nn.Linear(self.d_model, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with sensible defaults."""
        # Initialize embeddings
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.embed.padding_idx is not None:
            nn.init.constant_(self.embed.weight[self.embed.padding_idx], 0)

        # Initialize output projection
        nn.init.xavier_uniform_(self.out_linear.weight)
        nn.init.constant_(self.out_linear.bias, 0)

    def forward(self, inputs, lengths=None):
        """
        Optimized Mamba forward pass.
        We do NOT mask internally. We allow Mamba to process padding tokens
        naturally to preserve state dynamics. The training loop handles
        ignoring the resulting padding outputs.
        """
        # Embed tokens: (batch_size, seq_len, d_model)
        x = self.embed(inputs)
        x = self.dropout(x)

        # CRITICAL: Mamba kernels require contiguous memory layout
        x = x.contiguous()

        # Process through Mamba blocks
        for mamba_block, layer_norm in zip(self.mamba_blocks, self.layer_norms):
            # Pre-norm architecture
            residual = x
            x = layer_norm(x)

            # Block
            x = mamba_block(x)

            # Residual connection
            # NOTE: We removed "if mask is not None: x = x * mask"
            # to allow correct State Space Model evolution.
            x = residual + self.dropout(x)

        # Final normalization
        x = self.final_norm(x)

        # Project to logits: (batch_size, seq_len)
        scores = self.out_linear(x).squeeze(-1)

        return scores

    # ---
    # --- Adapter Methods (The "Contract") ---
    # ---

    def get_scores_for_batch(self, batch):
        """
        Adapter for training.
        Unpacks batch, calls `self.forward`, and returns all scores.

        Args:
            batch: The raw, collated batch from the DataLoader.
                   Typically [inputs, lengths, classification_targets]

        Returns:
            scores: torch.Tensor of shape (batch_size, seq_len)
            targets: torch.Tensor of shape (batch_size,)
        """
        # Unpack the batch
        inputs, lengths, classification_targets = batch

        # Move tensors to the model's device
        inputs = inputs.to(self.embed.weight.device)
        lengths = lengths.to(self.embed.weight.device)
        classification_targets = classification_targets.to(self.embed.weight.device)

        # Call forward pass
        scores = self.forward(inputs, lengths)

        # Return scores and targets
        return scores, classification_targets

    def get_final_scores(self, batch):
        """
        Adapter for evaluation.
        Unpacks batch, calls `self.forward`, and returns final logit.

        Args:
            batch: The raw, collated batch from the DataLoader.

        Returns:
            last_logits: torch.Tensor of shape (batch_size,)
                        The logit from the last real token for each item.
        """
        # Unpack the batch
        inputs, lengths, _ = batch

        # Move tensors to the model's device
        inputs = inputs.to(self.embed.weight.device)
        lengths = lengths.to(self.embed.weight.device)

        # Call forward pass
        # scores shape: (batch_size, seq_len)
        scores = self.forward(inputs, lengths)

        # Find the index of the last token for each sequence
        # Shape: (batch_size,)
        last_indices = (lengths - 1).long()

        # Gather the scores from the last valid position
        # Shape: (batch_size, 1) -> (batch_size,)
        last_logits = scores.gather(
            1, last_indices.unsqueeze(1)
        ).squeeze(1)

        return last_logits