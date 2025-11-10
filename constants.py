import torch

# --- Tokenizer ---
TOKENIZER_NAME = 'meta-llama/Llama-3.2-3B-Instruct' 
PAD_TOKEN = '[PAD]'

# --- Data Processing ---
VAL_SIZE = 400               
MAX_LEN = 200                 
MIN_SENTENCE_LENGTH = 3

# --- Training ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'