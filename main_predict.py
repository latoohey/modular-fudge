import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser
import numpy as np
import random
from tqdm import tqdm
import os

from models import get_model  # Import the factory ---
from constants import *
from util import num_params

def load_classifier(ckpt_path, device, args):
    """Loads a trained classifier from a checkpoint using the model factory."""
    print(f"Loading classifier from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Load args *from the checkpoint* to know what model to build
    model_args = checkpoint['args']
    
    # This assumes your main_train.py saved 'tokenizer_name' in its args
    if not hasattr(model_args, 'tokenizer_name'):
        raise ValueError("Checkpoint 'args' must contain 'tokenizer_name'!")
        
    classifier_tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name, 
        token=args.hf_token # Use token from CLI args
    )
    if classifier_tokenizer.pad_token is None:
        classifier_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    
    # Get vocab size for the model
    vocab_size = len(classifier_tokenizer)
    
    # --- Use the factory to build the correct model ---
    model = get_model(model_args, vocab_size)
    
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Classifier loaded (Type: {model_args.model_type}, Epochs: {checkpoint['epoch']}).")
    return model, classifier_tokenizer

def generate_guided(
    llm,
    llm_tokenizer,
    classifier,
    classifier_tokenizer,
    prompt,
    max_len,
    condition_lambda,
    top_k
):
    """
    Performs FUDGE-style guided generation for a single prompt.
    This function is now model-agnostic.
    """
    device = llm.device

    # --- Create the Llama 3.1 Chat Prompt ---
    messages = [
        {"role": "user", "content": prompt}
    ]

    # This applies the template and adds the "assistant" prompt
    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # We store the original prompt's length to slice it off later
    prompt_length = input_ids.shape[1]
    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_len):
            # --- A: Get Base LLM Logits ---
            llm_outputs = llm(generated_ids)
            # Get logits (which are float16) and cast them to float32
            next_token_logits = llm_outputs.logits[:, -1, :].float()

            # --- B: Get Top-K Candidates ---
            top_logits, top_indices = torch.topk(next_token_logits, top_k)

            # --- C: Run Classifier on Candidates ---

            # 1. Create k candidate prefixes
            candidate_prefixes = torch.cat(
                [generated_ids.expand(top_k, -1), top_indices.squeeze(0).unsqueeze(-1)],
                dim=-1
            )

            # 2. Get the lengths.
            # Since all candidate_prefixes are the same length (no padding),
            # we can create the lengths tensor directly.
            current_seq_len = candidate_prefixes.shape[1]
            lengths = torch.LongTensor([current_seq_len] * top_k).to(device)

            # 3. Create the batch directly from the token IDs
            #    We are skipping the decode/re-encode step!
            batch = [
                candidate_prefixes,  # Use the LLM's token IDs directly
                lengths,
                None
            ]

            # 4. Get classifier scores
            #    The 'get_final_scores' adapter works as-is.
            last_token_logits = classifier.get_final_scores(batch)

            # --- D: Combine and Select ---

            # Get LLM log probs: log(P(x))
            llm_log_probs = F.log_softmax(top_logits, dim=-1)

            # We need log(P(a|x)), which for a binary classifier
            # logit is log(sigmoid(logit)).
            # This gives the log-probability of the *positive class* (style=1).
            classifier_log_probs = F.logsigmoid(last_token_logits)

            # FUDGE: log(P(x|a)) = log(P(x)) + lambda * log(P(a|x))
            # Shape: (1, top_k) + (top_k,) -> (1, top_k)
            combined_log_probs = llm_log_probs + (condition_lambda * classifier_log_probs)

            best_token_index = torch.argmax(combined_log_probs)
            next_token_id = top_indices[0, best_token_index].unsqueeze(0)

            # --- E: Append and Yield/Repeat ---
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            # --- THIS IS THE NEW LOGIC ---
            # 1. Decode the single new token
            new_text = llm_tokenizer.decode(next_token_id.squeeze(0), skip_special_tokens=True)

            # 2. "Yield" it back to the caller
            yield new_text
            # --- END OF NEW LOGIC ---

            if next_token_id.item() == llm_tokenizer.eos_token_id:
                break

def main(args):
    # Set device and seeds
    device = torch.device(DEVICE)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load our trained classifier
    
    classifier, classifier_tokenizer = load_classifier(args.ckpt, device, args)
    
    # 2. Load the base LLM
    print(f"Loading base LLM: {args.llm_model}...")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        dtype=torch.float16, 
        token=args.hf_token # Pass token for gated models
    ).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model, 
        token=args.hf_token # Pass token for gated models
    )
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    
    # --- Check mode: File (Comprehensive) or Interactive (Boutique) ---

    if args.input_file:
        # --- File Mode (Comprehensive Test) ---
        print(f"Running comprehensive test on {args.input_file}...")
        prompts = []
        with open(args.input_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                prompts.append(line.strip())

        with open(args.output_file, 'w', encoding='utf-8') as wf:
            for prompt in tqdm(prompts, desc="Generating"):
                
                # 1. Call the generator
                output_generator = generate_guided(
                    llm, llm_tokenizer, classifier, classifier_tokenizer,
                    prompt, args.max_len, args.condition_lambda, args.top_k
                )
                
                # 2. Exhaust the generator into a single string
                full_output = "".join(list(output_generator))

                # 3. Write the full string
                # Remove the prompt from the output for a clean file
                output_only = full_output.strip() # Slicing removed, as we only yield new tokens
                wf.write(output_only + '\n')
        print(f"Batch generation complete. Output saved to {args.output_file}")

    else:
        # --- Interactive Mode (Boutique Test) ---
        print("Entering interactive mode...")
        while True:
            try:
                prompt = input("Enter a prompt (or 'q' to quit): ")
                if prompt.lower() == 'q':
                    break
                condition_lambda_str = input("Enter a lambda value: ")
                print(f"--- Generating with lambda={int(condition_lambda_str)} ---")

                output_generator = generate_guided(
                    llm,
                    llm_tokenizer,
                    classifier,
                    classifier_tokenizer,
                    prompt,
                    args.max_len,
                    int(condition_lambda_str),
                    args.top_k
                )
                for new_token in output_generator:
                    print(new_token, end="", flush=True)
                print("\n---") # Add a newline at the end

            except KeyboardInterrupt:
                print("\nExiting.")
                break


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # --- Required Arguments ---
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the *classifier* model_best.pth.tar file')
    parser.add_argument('--hf_token', type=str, required=True, 
                        help='HuggingFace Token (for Llama, etc.)')

    # --- Model Arguments ---
    parser.add_argument('--llm_model', type=str, default='gpt2',
                        help='The base LLM to guide (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")')
    

    parser.add_argument('--condition_lambda', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='generated_output.txt')
    parser.add_argument('--seed', type=int, default=24601, help='random seed')
    
    args = parser.parse_args()
    main(args)