
-----

# Modular Causal Classifier Training

This project provides a modular, model-agnostic pipeline for training causal sequence classifiers, and is built off the FUDGE: Controlled Text Generation With Future Discriminators (FUDGE) system from Yang and Klein. ([FUDGE GitHub](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation); [FUDGE Paper](https://arxiv.org/abs/2104.05218))

The core principle is that the training, evaluation, and data-loading logic are completely decoupled from the specific model architecture. You can add, remove, or experiment with different model types (e.g., LSTM, Mamba, Transformer Decoder) by simply adding a new "model file" without changing the main training or evaluation scripts.

## How it Works: The Modular Architecture

The system is designed with training scripts and interchangeable models.

  * **Training Scripts:** `main_train.py` and `evaluate.py`.:

      * Loading data via `data.py`.
      * Looping through epochs and batches.
      * Calculating the "implicit prefix" loss.
      * Saving checkpoints.
      * Running the final evaluation.

  * **Models:** Files within the `models/` directory (e.g., `models/lstm.py`). Each file defines a specific model implementation.

  * **The "Contract":**
    For the scripts to recognize a new model, the model file *must* provide a class that fulfills a specific "contract." This contract is a set of required methods that the training/evaluation scripts call.

    Any model file you add **must** provide the following three methods:

    1.  `__init__(self, args, vocab_size)`: Initializes the model, taking the master `args` and the `vocab_size` from the tokenizer.
    2.  `get_scores_for_batch(self, batch)`: An "adapter" method for training. It takes the raw batch from the DataLoader, runs its internal `forward` pass, and returns all per-token scores.
    3.  `get_final_scores(self, batch)`: An "adapter" method for evaluation. It takes the raw batch and returns only the *single final logit* for the last token of each sequence.

## Project Structure

```
project-directory/
|
|-- main_train.py         # Main training script
|-- evaluate.py           # Evaluation script
|-- data.py               # Data loading and preprocessing
|-- util.py               # Helper functions (e.g., pad_mask)
|-- create_data_files.py  # Takes dataset files and splits them into test and train directories
|-- main_predict.py       # Script to run classifier with LLM and view results
|
|-- models/
|   |-- __init__.py     # The "model factory"
|   |-- lstm.py         # LSTM model definition
|
|-- trained_models/
|   |-- model_best.pth.tar  # Output checkpoints
|
|-- data/
|   |-- splits
|       |-- train/
|       |-- test/
```

## 1\. Setup

### Dependencies

Install the required packages.

```bash
pip install torch transformers scikit-learn
```

### Data Setup

The `data.py` script expects a specific folder structure. Your `data_dir` must contain `train` and `test` folders.

For a binary classifier, this would be:

  * `data/splits/train/style_0.txt`
  * `data/splits/train/style_1.txt`
  * `data/splits/test/style_0.txt`
  * `data/splits/test/style_1.txt`

*(Note: The `data.py` loader can be edited to match your specific filenames, e.g., `holmes.txt`).*

## 2\. How to Use

### Training a Model

Use the `main_train.py` script. The `--model_type` argument tells the factory which "engine" to load from the `models/` directory.

You must also provide model-specific hyperparameters.

```bash
python main_train.py \
    --data_dir data \
    --save_dir trained_models \
    --hf_token YOUR_HF_TOKEN \
    \
    --model_type lstm \
    --lstm_hidden_dim 300 \
    --lstm_num_layers 3 \
    \
    --batch_size 32 \
    --epochs 20 \
    --lr 1e-4 \
    --seed 24601 \
    --pos_cat holmes \
    --neg_cat encyclopedia
```

### Evaluating a Model

Use the `evaluate.py` script. This script will automatically load the correct model architecture (e.g., LSTM) by reading the `args` saved *inside* the checkpoint file.

```bash
python evaluate.py \
    --ckpt ./trained_models/model_best.pth.tar \
    --data_dir ./data \
    --hf_token YOUR_HF_TOKEN \
    --batch_size 32
```

## 3\. How to Extend (Adding a New Model)

Here is how to add a new `MyTransformer` model.

**Step 1: Create `models/my_transformer.py`**
Copy `models/lstm.py` as a template. Inside the new file, define your new class (`MyTransformerClassifier`).

**Step 2: Implement the "Contract"**
Modify the class to fulfill the three required methods:

1.  **`__init__(self, args, vocab_size)`:**

      * Change the layers from `nn.LSTM` to your new `nn.TransformerDecoderLayer`.
      * Read your new hyperparameters from the `args` object (e.g., `args.transformer_n_heads`).

2.  **`get_scores_for_batch(self, batch)`:**

      * Unpack the `batch` as needed.
      * Call your model's internal `forward` method.
      * Return the `(scores, classification_targets)` tuple, where `scores` has the shape `(batch_size, seq_len)`.

3.  **`get_final_scores(self, batch)`:**

      * Unpack the `batch`.
      * Call your `forward` method.
      * Find the logit for the *last unpadded token* for each item.
      * Return the `last_logits` tensor with shape `(batch_size,)`.

**Step 3: Update the Factory**
Open `models/__init__.py` and add your new model to the `get_model` function:

```python
from .lstm import LSTMClassifier
from .my_transformer import MyTransformerClassifier  # <-- 1. Import

def get_model(args, vocab_size):
    if args.model_type == 'lstm':
        return LSTMClassifier(args, vocab_size)
    elif args.model_type == 'my_transformer':  # <-- 2. Add selector
        return MyTransformerClassifier(args, vocab_size)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
```

You can now train your new model by passing `--model_type my_transformer` to `main_train.py`. The training and evaluation scripts will work without any other changes.


## 4. Running Guided Generation (`main_predict.py`)

This script is the "driver" that uses your trained classifier. It loads a base LLM (like Llama 3.1) and your trained classifier checkpoint. It uses the same `models` factory to ensure the correct model architecture is loaded from the checkpoint. It then performs FUDGE-style guided generation by getting the LLM's top-k next tokens, re-ranking them using your classifier's "style" scores, and selecting the new best token. The script can be run in an interactive "chat" mode for boutique testing or in a "file" mode to batch-process a list of prompts.