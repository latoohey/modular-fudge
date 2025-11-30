pip install textstat
pip install transformers
pip install sentence_transformers
pip install nltk

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nltk
import sys
from pathlib import Path
import textstat
from collections import Counter

# --- 1. NLTK Setup ---
def ensure_nltk_resources():
    resources = ['punkt_tab', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            print(f"Downloading missing NLTK resource: {res}...")
            nltk.download(res, quiet=True)

# --- 3. The Evaluator Class ---
class StyleTransferEvaluator:
    def __init__(self, style_model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Evaluator on {self.device}...")

        self.style_model_path = style_model_path
        self._load_models()

    def _load_models(self):
        # 1. Fluency (GPT-2 XL)
        print("Loading GPT-2 XL (Fluency)...")
        self.ppl_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        self.ppl_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(self.device)
        self.ppl_model.eval()

        # 2. Coherence (SBERT)
        print("Loading SBERT (Coherence)...")
        self.coherence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # 3. Style Classifier (Custom)
        print(f"Loading Style Classifier from {self.style_model_path}...")
        self.style_tokenizer = AutoTokenizer.from_pretrained(self.style_model_path, local_files_only=True)
        self.style_model = AutoModelForSequenceClassification.from_pretrained(
            self.style_model_path, local_files_only=True
        ).to(self.device)
        self.style_model.eval()

    # --- Calculation Methods ---

    def _calc_perplexity(self, text):
        if not text or not text.strip(): return 0.0
        encodings = self.ppl_tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)
        if input_ids.size(1) > 1024: input_ids = input_ids[:, :1024]

        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            return torch.exp(outputs.loss).item()

    def _calc_style_score(self, text):
        if not text: return 0.0
        inputs = self.style_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.style_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs[0][1].item()

    def _calc_coherence(self, text):
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2: return 0.0
        embeddings = self.coherence_model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)
        # Average adjacent scores
        adjacent_scores = [cosine_scores[i][i+1].item() for i in range(len(sentences)-1)]
        return np.mean(adjacent_scores).item()

    def _calc_distinct_n(self, text, n):
        tokens = text.lower().split()
        if len(tokens) < n: return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if not ngrams: return 0.0
        return len(set(ngrams)) / len(ngrams)

    def _calc_stylometrics(self, text):
        # Default safe return
        defaults = {k: 0.0 for k in METRIC_CONFIG.keys() if k not in ['perplexity', 'coherence', 'style_score', 'dist_1', 'dist_2', 'dist_3', 'token_length', 'elapsed_time']}
        if not text or not text.strip(): return defaults

        sentences = nltk.sent_tokenize(text)
        words = []
        for sent in sentences:
            words.extend(nltk.word_tokenize(sent.lower()))
        words_alpha = [w for w in words if w.isalpha()] or ['placeholder']
        sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]

        # Passive Voice Logic (Preserved exactly)
        passive_indicators = {'was', 'were', 'been', 'being', 'be'}
        passive_count = 0
        for sent in sentences:
            sent_words = nltk.word_tokenize(sent.lower())
            for i in range(len(sent_words) - 1):
                if sent_words[i] in passive_indicators:
                    if sent_words[i+1].endswith('ed'):
                        passive_count += 1
                        break

        return {
            'avg_sent_length': np.mean(sent_lengths).item(),
            'avg_word_length': np.mean([len(w) for w in words_alpha]).item(),
            'type_token_ratio': len(set(words_alpha)) / len(words_alpha),
            'long_word_ratio': sum(1 for w in words_alpha if len(w) > 6) / len(words_alpha),
            'flesch_reading_ease': textstat.flesch_reading_ease(text) if textstat.flesch_reading_ease(text) is not None else 0.0,
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text) if textstat.flesch_kincaid_grade(text) is not None else 0.0,
            'complex_sentence_ratio': sum(1 for l in sent_lengths if l > 20) / len(sentences),
            'passive_voice_ratio': passive_count / len(sentences)
        }

    # --- Core Processing Methods ---

    def evaluate_text(self, input_data):
        """
        Accepts either a string or a dict (row). Returns a dict of metrics.
        """
        # Handle input types
        if isinstance(input_data, dict):
            text = input_data.get('output', "")
            result = input_data.copy()
        else:
            text = input_data
            result = {'output': text}

        # Calculate FUDGE metrics
        result['perplexity'] = self._calc_perplexity(text)
        result['coherence'] = self._calc_coherence(text)
        result['dist_1'] = self._calc_distinct_n(text, 1)
        result['dist_2'] = self._calc_distinct_n(text, 2)
        result['dist_3'] = self._calc_distinct_n(text, 3)
        result['style_score'] = self._calc_style_score(text)
        result['token_length'] = len(self.style_tokenizer.encode(text))

        # Calculate Stylometric metrics
        stylometrics = self._calc_stylometrics(text)
        result.update(stylometrics)

        return result

    def evaluate_file(self, input_path, output_path, treatments=['model_name', 'top_k', 'lambda']):
        """
        Reads CSV, processes rows, prints reports, saves CSV.
        """
        df = pd.read_csv(input_path)
        print(f"Processing {len(df)} rows from {input_path}...")


#        with open('splits/test/eb.txt', 'r', encoding='utf-8') as f:
#            lines = [line.strip() for line in f]
#        # Construct the dictionary first
#        data = {
#            'model_name': 'eb',
#            'top_k': 100,
#            'lambda': 0,
#            'prompt': 'Prompt',
#            'elapsed_time': 1,
#            'output': lines
#        }
#        # Create the DataFrame
#        df = pd.DataFrame(data)

        results = []
        for record in tqdm(df.to_dict('records')):
            results.append(self.evaluate_text(record))

        result_df = pd.DataFrame(results)

        # Generate Reports
        print("\n=== Evaluation Summary ===")
        print_metrics_summary(result_df, treatments)

        if 'model_name' in result_df.columns and 'baseline' in result_df['model_name'].values:
            print("\n=== Style Movement (vs Baseline) ===")
            print_style_movement(result_df)

        result_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
        return result_df

def print_metrics_summary(df, treatments):
    # Identify which metrics exist in this dataframe
    available_metrics = [m for m in METRIC_CONFIG.keys() if m in df.columns]

    # Group by treatments
    try:
        summary = df.groupby(treatments)[available_metrics].mean().round(2)
    except KeyError:
        # Fallback if treatments columns missing
        summary = df[available_metrics].mean().to_frame().T.round(2)

    print(summary)
    print("-" * 60)
    print("🏆 BEST PERFORMERS")
    print("-" * 60)

    for metric in available_metrics:
        higher_is_better = METRIC_CONFIG.get(metric, True)

        # Find best index
        if higher_is_better:
            best_idx = summary[metric].idxmax()
            direction = "High"
        else:
            best_idx = summary[metric].idxmin()
            direction = "Low"

        best_val = summary.loc[best_idx, metric]

        # Format label
        label = ", ".join(map(str, best_idx)) if isinstance(best_idx, tuple) else str(best_idx)
        print(f"• {metric:<22} ({direction}): {label} ({best_val})")

def print_style_movement(df):
    baseline_df = df[df['model_name'] == 'baseline']
    if baseline_df.empty: return

    for model in df['model_name'].unique():
        if model == 'baseline': continue

        model_df = df[df['model_name'] == model]
        print(f"\n{model} vs Baseline:")

        for metric, expected_dir in STYLE_MOVEMENT_CHECKS:
            if metric not in df.columns: continue

            base_mean = baseline_df[metric].mean()
            model_mean = model_df[metric].mean()
            change = model_mean - base_mean
            pct = (change / base_mean * 100) if base_mean != 0 else 0

            is_good = (expected_dir == 'increase' and change > 0) or \
                    (expected_dir == 'decrease' and change < 0)
            symbol = "✓" if is_good else "✗"

            print(f"  {metric}: {base_mean:.2f} → {model_mean:.2f} ({pct:+.1f}%) {symbol}")
            
def merge_evaluation_reports(project_path):
    """Finds all eval_* CSVs, merges them, and generates a master summary."""
    folder = Path(project_path)
    csv_files = list(folder.glob('eval_*.csv'))

    if not csv_files:
        print("No eval files found to merge.")
        return

    print(f"Merging {len(csv_files)} files...")
    df_list = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(df_list, ignore_index=True).drop_duplicates()

    print(f"Final dataset shape: {combined.shape}")

    # Run the report on the combined data
    print_metrics_summary(combined, ['model_name', 'top_k', 'lambda'])

    output_file = folder / 'combined_evals.csv'
    combined.to_csv(output_file, index=False)
    print(f"Combined report saved to {output_file}")
    
def mount_colab_helper(project_path, judge_path=None):
    """Helper to handle the specific pathing needs of Colab."""
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        base = '/content/drive/My Drive/'

        # Return updated paths
        p_path = f"{base}{project_path}"
        j_path = f"{base}{judge_path}" if judge_path else None
        return p_path, j_path
    return project_path, judge_path
            

if __name__ == '__main__':
    
    # --- 2. Configuration & Constants ---
    # defined in one place to ensure consistency across all printing/saving functions
    METRIC_CONFIG = {
        # Metric Name             # Higher is Better? (True/False)
        'perplexity':             False,
        'coherence':              True,
        'dist_1':                 True,
        'dist_2':                 True,
        'dist_3':                 True,
        'style_score':            True,
        'token_length':           True,  # Assuming preference for length, change to False if brevity preferred
        'avg_sent_length':        True,  # For formal style transfer, usually higher
        'avg_word_length':        True,
        'type_token_ratio':       True,
        'long_word_ratio':        True,
        'flesch_reading_ease':    False, # Lower = more complex/formal
        'flesch_kincaid_grade':   True,  # Higher = more complex
        'complex_sentence_ratio': True,
        'passive_voice_ratio':    False, # Context dependent, but usually minimize in active writing
        'elapsed_time':           False
    }

    STYLE_MOVEMENT_CHECKS = [
        ('avg_sent_length', 'increase'),
        ('avg_word_length', 'increase'),
        ('long_word_ratio', 'increase'),
        ('flesch_reading_ease', 'decrease'),
        ('complex_sentence_ratio', 'increase'),
    ]

    PROJECT_PATH_IN = "modular-fudge/data/test_results"
    JUDGE_PATH_IN = "modular-fudge/judge_classifier"
    DATA_FILE = "llm_baseline_tests.csv"
    SINGLE_TEXT =  """Geology is the scientific study of the Earth's physical structure, composition, and processes that shape its surface. It encompasses various disciplines, including physics, chemistry, biology, and environmental science, to understand the complex interactions between the Earth's lithosphere (rock), hydrosphere (water), atmosphere (air), and biosphere (living organisms)."""
    SINGLE_TEXT = """Strawberries are a popular and tasty fruit. They are typically red and have a sweet flavor that many people enjoy. Unlike some other fruits, strawberries have tiny seeds on their outer surface. Because of their good taste, people eat them in many ways, such as fresh, in a fruit salad, or made into jams or jellies."""

    ensure_nltk_resources()
    
    full_project_path, full_judge_path = mount_colab_helper(PROJECT_PATH_IN, JUDGE_PATH_IN)
    evaluator = StyleTransferEvaluator(full_judge_path)
    
    # OPTION A: Batch Process a CSV
    # evaluator.evaluate_file(
    #  input_path=f"{full_project_path}/{DATA_FILE}",
    #  output_path=f"{full_project_path}/eval_{DATA_FILE}")
    # print('Evaluation Complete')
    # OPTION B: Single Text
    # result = evaluator.evaluate_text(SINGLE_TEXT)
    # print("=== Evaluation Summary ===")
    # for key, value in result.items():
    #   if key != 'output':
    #     print(f"{key}: {value:.3f}")

    # OPTION C: Merge Files
    merge_evaluation_reports(full_project_path)