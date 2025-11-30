from argparse import Namespace
import csv
from google.colab import userdata
from huggingface_hub import login
from IPython.display import HTML, display
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

import os
import csv
import time
import pandas as pd
import itertools
from pathlib import Path
import gc

from models import get_model  # Import the factory ---
from constants import *
from util import num_params

# 1. Add the decorator. maxsize=1 is usually enough if you just
#    want to hold the current model in memory.
@lru_cache(maxsize=1)
def load_classifier(ckpt_path, device):
    """Loads a trained classifier from a checkpoint using the model factory."""

    # This print statement will only run the FIRST time you call the function
    # with a specific path/device combination.
    print(f"Loading classifier from {ckpt_path}...")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load args *from the checkpoint* to know what model to build
    model_args = checkpoint['args']
    print(f"Checkpoint args: {model_args}")

    # This assumes your main_train.py saved 'tokenizer_name' in its args
    if not hasattr(model_args, 'tokenizer_name'):
        tokenizer_name = CLASSIFIER_TOKENIZER_NAME # Ensure this global is defined or passed in
    else:
        tokenizer_name = model_args.tokenizer_name

    classifier_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Add pad token if it doesn't exist
    if classifier_tokenizer.pad_token is None:
        # 1. Check for Llama 3 specific fine-tune token
        if '<|finetune_right_pad_id|>' in classifier_tokenizer.get_vocab():
            classifier_tokenizer.pad_token = '<|finetune_right_pad_id|>'

        # 2. Check for generic reserved tokens (common in TikToken)
        elif '<|reserved_special_token_0|>' in classifier_tokenizer.get_vocab():
            classifier_tokenizer.pad_token = '<|reserved_special_token_0|>'

        # 3. Safe Fallback: Use EOS token (No resizing required)
        else:
            print("Warning: No dedicated pad token found. Using EOS token as PAD.")
            classifier_tokenizer.pad_token = classifier_tokenizer.eos_token

    vocab_size = len(classifier_tokenizer)
    print(f"Classifier vocab size: {vocab_size}")
    pad_token_id = classifier_tokenizer.pad_token_id

    # --- Use the factory to build the correct model ---
    # Ensure get_model is imported or defined in this scope
    model = get_model(model_args, vocab_size, pad_token_id)

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Classifier loaded (Type: {model_args.model_type}, Epochs: {checkpoint['epoch']}).")

    # Returns the tuple. The cache will store this entire tuple.
    return model, classifier_tokenizer

def calculate_combined_scores(top_logits, last_token_logits, condition_lambda, use_z_score=False):
    """
    Normalizes and combines LLM logits with Classifier scores.
    Returns: combined_log_probs (for selection), final_classifier_scores (for logging), llm_log_probs
    """
    # 1. Normalize LLM scores to log probs
    llm_log_probs = F.log_softmax(top_logits, dim=-1)

    # --- CHANGE 1: EARLY EXIT FOR OPTIMIZATION ---
    # If the classifier was skipped (lambda=0), return pure LLM scores immediately.
    if last_token_logits is None:
        # Create dummy zeros for the "classifier scores" so the logger doesn't crash.
        # We make it match the shape of top_logits [1, top_k]
        dummy_classifier_scores = torch.zeros_like(top_logits)

        # Return: (Pure LLM Scores, Dummy Zeros, Pure LLM Scores)
        return llm_log_probs, dummy_classifier_scores, llm_log_probs

    # 2. Normalize Classifier scores to log probs
    classifier_log_probs = F.log_softmax(last_token_logits, dim=-1)

    # Extract the "True" class score (assuming binary classification index 1 is target)
    if len(classifier_log_probs.shape) > 1 and classifier_log_probs.shape[-1] > 1:
        relevant_classifier_scores = classifier_log_probs[:, 1]
    else:
        relevant_classifier_scores = classifier_log_probs

    # 3. Apply Strategy
    if use_z_score:
        # Calculate stats across the top_k candidates
        c_mean = relevant_classifier_scores.mean()

        # --- CHANGE 2: FIX THE STD() CRASH ---
        # unbiased=False prevents crash when top_k=1 (div by zero error)
        c_std = relevant_classifier_scores.std(unbiased=False)

        if c_std < 1e-8: c_std = 1.0 # Safety

        final_classifier_scores = (relevant_classifier_scores - c_mean) / c_std
    else:
        final_classifier_scores = relevant_classifier_scores

    # 4. Combine: LLM_Log_Prob + (Lambda * Classifier_Score)
    combined_log_probs = llm_log_probs + (condition_lambda * final_classifier_scores)

    return combined_log_probs, final_classifier_scores, llm_log_probs

def select_next_token(combined_log_probs, top_indices, strategy="greedy", temperature=1.0):
    """
    Selects the next token index based on strategy.
    Returns: next_token_id (tensor), best_index_relative (int index of top_k)
    """
    if strategy == "sample":
        # Divide by temp to control randomness
        probs = F.softmax(combined_log_probs / temperature, dim=-1)
        # Sample from the distribution
        best_index_relative = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy default (Returns a 0-dim scalar tensor)
        best_index_relative = torch.argmax(combined_log_probs)

    # --- THE FIX IS HERE ---
    # We force convert to Python int regardless of dimensions.
    # argmax returns 0-dim, multinomial returns 2-dim. .item() handles both.
    if isinstance(best_index_relative, torch.Tensor):
        best_index_relative = int(best_index_relative.item())

    # Extract the actual token ID from the top_k list
    next_token_id = top_indices[0, best_index_relative].unsqueeze(0)

    return next_token_id, best_index_relative

def record_evaluation(evaluation_history, step, generated_ids, tokenizer,
                      top_indices, llm_scores, clf_scores, combined_scores, selected_idx):
    """
    Logs the step details to the history list.
    """
    current_context_str = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    top_k = top_indices.shape[1]

    step_data = {
        "step": step,
        "context": current_context_str,
        "candidates": []
    }

    # FIX: Normalize clf_scores to be 1D so we can loop over it easily
    # If it came from zeros_like(top_logits), it's [1, K]. We want [K].
    if clf_scores.dim() > 1:
        clf_scores = clf_scores.squeeze(0)

    for i in range(top_k):
        cand_id = top_indices[0, i].item()
        cand_token = tokenizer.decode([cand_id])

        # Safe extraction of scalar values
        s_llm = llm_scores[0, i].item()

        # FIX: Now we can safely use [i] for both Normal and Optimized cases
        s_clf = clf_scores[i].item()

        s_comb = combined_scores[0, i].item()
        is_winner = (i == selected_idx)

        step_data["candidates"].append({
            "token_text": cand_token,
            "llm_score": round(s_llm, 4),
            "classifier_score": round(s_clf, 4),
            "weighted_combined": round(s_comb, 4),
            "selected": is_winner
        })

    evaluation_history.append(step_data)
    
    
def generate_guided(
    llm,
    llm_tokenizer,
    classifier,
    classifier_tokenizer,
    prompt,
    max_len,
    condition_lambda,
    top_k,
    evaluation_history=None,
    use_z_score=False,
    strategy="greedy",
    temperature=1.0
):
    device = llm.device

    # ... (Steps 1, 2, and 3: Template, Sanitization, Tokenization remain same) ...
    # [Pasted for context]
    try:
        if callable(CUSTOM_PROMPT_TEMPLATE):
            messages = CUSTOM_PROMPT_TEMPLATE(prompt)
        else:
            messages = prompt
    except NameError:
        messages = prompt

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], str):
        messages = [{"role": "user", "content": messages[0]}]

    add_gen_prompt = globals().get('ADD_GENERATION_PROMPT', True)
    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_gen_prompt,
        return_tensors="pt"
    ).to(device)

    generated_ids = input_ids

    with torch.no_grad():
        for step in range(max_len):
            # --- A: Get Base LLM Logits ---
            llm_outputs = llm(generated_ids)
            next_token_logits = llm_outputs.logits[:, -1, :].float()

            # --- B: Get Top-K Candidates ---
            top_logits, top_indices = torch.topk(next_token_logits, top_k)

            # --- C: Run Classifier (OPTIMIZED) ---

            # If lambda is effectively zero (smaller than 0.000001), skip the heavy lift
            # Check if we are effectively turning the classifier off
            if abs(condition_lambda) < 1e-6:
                last_token_logits = None
            else:
                # Only do this heavy VRAM expansion if we actually plan to use it

                # Create sequences: [Current Context + Candidate Token]
                candidate_prefixes = torch.cat(
                    [generated_ids.expand(top_k, -1), top_indices.squeeze(0).unsqueeze(-1)],
                    dim=-1
                )

                # Prepare classifier batch
                current_seq_len = candidate_prefixes.shape[1]
                lengths = torch.LongTensor([current_seq_len] * top_k).to(device)
                batch = [candidate_prefixes, lengths, None]

                # Get raw classifier scores
                last_token_logits = classifier.get_final_scores(batch)

            # --- D: Calculate Scores (Helper 1) ---
            # If lambda is 0, this calculates: LLM_Score + (0 * 0) = LLM_Score
            combined_scores, clf_scores, llm_log_probs = calculate_combined_scores(
                top_logits,
                last_token_logits,
                condition_lambda,
                use_z_score
            )

            # --- E: Select Token (Helper 2) ---
            next_token_id, best_idx = select_next_token(
                combined_scores,
                top_indices,
                strategy=strategy,
                temperature=temperature
            )

            # --- F: Log (Helper 3) ---
            if evaluation_history is not None:
                record_evaluation(
                    evaluation_history, step, generated_ids, llm_tokenizer,
                    top_indices, llm_log_probs, clf_scores, combined_scores, best_idx
                )

            # --- G: Append and Yield ---
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
            new_text = llm_tokenizer.decode(next_token_id.squeeze(0), skip_special_tokens=True)

            yield new_text

            if next_token_id.item() == llm_tokenizer.eos_token_id:
                break
            
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
    
    
def setup():
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)

    if SAVE_TESTS_TO_DRIVE or CLASSIFIER_PATH is not None:
        from google.colab import drive
        drive.mount('/content/drive')

    def get_classifier(classifier_model_name, device):
        model_checkpoint = f'{classifier_model_name}.pth.tar'

        release_url = None
        if GITHUB_RELEASE_VERSION is not None:
            release_url = f"https://github.com/latoohey/modular-fudge/releases/download/{GITHUB_RELEASE_VERSION}/{model_checkpoint}"

            # --- LOGIC ADDED HERE ---
            if os.path.exists(model_checkpoint):
                print(f"Found {model_checkpoint} locally. Skipping download.")
            else:
                print(f"Downloading {model_checkpoint}...")
                !wget "{release_url}" -O {model_checkpoint}
                print("Model downloaded")

            ckpt_path = model_checkpoint

        else:
            ckpt_path = os.path.join(CLASSIFIER_PATH, model_checkpoint)
            print("Using Drive model")

        # 1. Load our trained LSTM classifier
        classifier, classifier_tokenizer = load_classifier(ckpt_path, device)
        print("--- Classifier loaded! ---")
        return classifier, classifier_tokenizer

    def main():
        seed_everything(SEED)

        setup()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        llm_model = LLM_MODEL_NAME
        print(f"Loading base LLM: {llm_model}...")
        llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            dtype=torch.float16
        ).to(device)
        llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_TOKENIZER_NAME
        )
        # Add pad token if it doesn't exist
        if llm_tokenizer.pad_token is None:
            # 1. Check for Llama 3 specific fine-tune token
            if '<|finetune_right_pad_id|>' in llm_tokenizer.get_vocab():
                llm_tokenizer.pad_token = '<|finetune_right_pad_id|>'

            # 2. Check for generic reserved tokens (common in TikToken)
            elif '<|reserved_special_token_0|>' in llm_tokenizer.get_vocab():
                llm_tokenizer.pad_token = '<|reserved_special_token_0|>'

            # 3. Safe Fallback: Use EOS token (No resizing required)
            else:
                print("Warning: No dedicated pad token found. Using EOS token as PAD.")
                llm_tokenizer.pad_token = llm_tokenizer.eos_token

        print("--- LLM loaded! ---")

        model_defs = {
            "llm": llm,
            "llm_tokenizer": llm_tokenizer,
            "classifier_model_name": CLASSIFIER_MODEL_NAME,
            "device": device
        }


        if (TESTING_TYPE=="prompted"):
            prompted_testing(model_defs)
        elif (TESTING_TYPE=="targeted"):
            targeted_testing(model_defs)
        elif (TESTING_TYPE=="grid"):
            grid_testing(model_defs)
        elif (TESTING_TYPE=="token_eval"):
            token_evaluation_testing(model_defs, TESTING_PROMPT)

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    
    import os
    from google.colab import drive

    # 1. Mount Google Drive
    drive.mount('/content/drive')

    # 2. Define the directory on your Drive to store the wheels
    #    We use a specific folder name to keep it organized.
    wheel_dir = '/content/drive/MyDrive/colab_wheels/mamba_builds'
    os.makedirs(wheel_dir, exist_ok=True)

    # 3. Define the package versions you want
    packages = [
        "causal-conv1d>=1.4.0",
        "mamba-ssm"
    ]

    # 4. Check if wheels already exist in your Drive
    print(f"Checking for existing wheels in {wheel_dir}...")
    existing_wheels = [f for f in os.listdir(wheel_dir) if f.endswith('.whl')]

    if len(existing_wheels) >= len(packages):
        print("✅ Found pre-built wheels! Installing from Drive...")
        # Install directly from your Drive folder
        pip install "$wheel_dir"/*.whl
    else:
        print("⚠️ No wheels found. Building from source (this will take time once)...")

        # Install build dependencies first
        pip install packaging ninja

        # Build the wheels and save them directly to your Drive
        # We use --no-deps to avoid building wheels for huge packages like PyTorch
        print(f"Building wheels to {wheel_dir}...")
        pip wheel {" ".join(packages)} --wheel-dir="$wheel_dir" --no-deps

        # Now install the newly built wheels
        print("Installing newly built wheels...")
        pip install "$wheel_dir"/*.whl

    print("🎉 Done! Mamba and Causal-Conv1d are ready.")
    parser = ArgumentParser()
    
    # --- Required Arguments ---
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the *classifier* model_best.pth.tar file')
    parser.add_argument('--hf_token', type=str, required=True, 
                        help='HuggingFace Token (for Llama, etc.)')

    # --- Model Arguments ---
    parser.add_argument('--llm_model', type=str, default='gpt2',
                        help='The base LLM to guide (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")')
    
    TESTING_TYPE = "token_eval" # "grid" or "targeted" or "prompted" or "token_eval"
    # TESTING_PROMPT is needed if TESTING_TYPE is "targeted" or "token_eval"
    TESTING_PROMPT = "write a paragraph about north america"

    GRID_TEST_RUN_NAME = "mamba_128_4_16_1"
    SAVE_TESTS_TO_DRIVE = True
    TEST_PROMPTS_FILE_PATH = "modular-fudge/data/eval_prompts.csv"
    PROMPTS_TO_TEST_LIMIT = False
    GRID_LAMBDAS = [1.4]
    GRID_CLASSIFIER_NAMES = ['mamba_128_4_16_1']
    GRID_TOP_KS = [100]
    GRID_USE_Z_SCORES = [True]

    #TESTING_LAMBDA is needed if TESTING_TYPE is "token_eval"
    TESTING_LAMBDA=0

    SEED = 24601

    CLASSIFIER_MODEL_NAME = "lstm_2_256"

    GITHUB_RELEASE_VERSION = "v2.0"
    #---OR---
    CLASSIFIER_PATH = '' # '/content/drive/MyDrive/modular-fudge/trained_models'

    KEEP_EVALUATION_HISTORY = True
    EVAL_STEP_TO_ANALYZE = 11

    USE_Z_SCORE = True
    STRATEGY = "greedy"  # Options: "greedy", "sample"
    TEMPERATURE = 1.0     # Only used if strategy="sample"

    CLASSIFIER_TOKENIZER_NAME = 'meta-llama/Llama-3.2-3B-Instruct'

    # (You will need to accept the license on Hugging Face first for Llama)
    # LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    # LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

    LLM_TOKENIZER_NAME = 'meta-llama/Llama-3.2-3B-Instruct'

    # Keep both of these values low if running "token_eval"
    MAX_NEW_TOKENS = 512
    TOP_K = 100

    # PROMPT TEMPLATE
    # To use just a plain prompt, set this to None:
    # CUSTOM_PROMPT_TEMPLATE = None

    # --- OR ---

    # To use a custom template, define a lambda function.
    # The lambda MUST accept one argument (e.g., 'p') which will be your prompt string.
    # It MUST return the 'messages' list structure you want.

    # Example 1: Add a simple prefix
    # CUSTOM_PROMPT_TEMPLATE = lambda p: [
    #     {"role": "user", "content": f"Task: Answer the following question. {p}"}
    # ]

    # Example 2: Add a System Prompt
    # This one is the base format for LLama models
    CUSTOM_PROMPT_TEMPLATE = lambda p: [{"role": "user", "content": p}]
    ADD_GENERATION_PROMPT = True

    parser.add_argument('--condition_lambda', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='generated_output.txt')
    parser.add_argument('--seed', type=int, default=24601, help='random seed')
    
    args = parser.parse_args()
    main(args)