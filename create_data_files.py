import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
# 1. Set your input file paths
FILE_1_PATH = 'data\datasets\sampled_encyclopedia_chunks.csv'
FILE_2_PATH = 'data\datasets\sampled_holmes_chunks.csv'

FILE_1_OUTPUT_NAME = 'encyclopedia'
FILE_2_OUTPUT_NAME = 'holmes'

DATA_PATH = 'data\splits'  # Directory to save output files

# 2. Set the name of the column containing your text
TEXT_COLUMN_NAME = 'text'


# 3. Set your sampling and split parameters
NUM_SAMPLES = 1500
TEST_SPLIT_SIZE = 0.20  # 80% train, 20% test
RANDOM_SEED = 24601      # Ensures your splits are reproducible
# ---------------------

def process_csv(input_path, output_name, text_col, n_samples, test_size, seed):
    """
    Reads a CSV, samples it, splits it, and writes train/test .txt files.
    """
    print(f"--- Processing {input_path} ---")
    
    try:
        # Read *only* the required text column to save memory
        df = pd.read_csv(input_path, usecols=[text_col])
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}. Skipping.")
        return
    except ValueError as e:
        # This catches if the 'text' column doesn't exist
        print(f"Error reading {input_path}: {e}. Make sure '{text_col}' column exists. Skipping.")
        return
    except Exception as e:
        print(f"An unexpected error occurred reading {input_path}: {e}. Skipping.")
        return

    # Handle files with fewer rows than desired samples
    n_rows = len(df)
    if n_rows < n_samples:
        print(f"Warning: {input_path} only has {n_rows} rows. Using all of them.")
        actual_samples = n_rows
    else:
        actual_samples = n_samples

    # Get the random sample of text data
    # We drop any rows where the text might be missing (NaN)
    sampled_data = df.dropna(subset=[text_col]).sample(
        n=actual_samples, 
        random_state=seed
    )

    # Split the data into train and test sets
    train_texts, test_texts = train_test_split(
        sampled_data[text_col],
        test_size=test_size,
        random_state=seed
    )

    # Define output filenames
    file = f"{output_name}.txt"
    
    train_file_path = os.path.join(DATA_PATH, 'train', file)
    test_file_path = os.path.join(DATA_PATH, 'test', file)

    # Write the train file
    try:
        with open(train_file_path, 'w', encoding='utf-8') as f:
            for line in train_texts:
                f.write(str(line) + '\n')
        print(f"Created {file} at {train_file_path} with {len(train_texts)} lines.")

        # Write the test file
        with open(test_file_path, 'w', encoding='utf-8') as f:
            for line in test_texts:
                f.write(str(line) + '\n')
        print(f"Created {file} at {test_file_path} with {len(test_texts)} lines.")
        
    except IOError as e:
        print(f"Error writing file: {e}")
    
    print(f"--- Finished {input_path} ---\n")

# --- Main execution ---
if __name__ == "__main__":
    
    # Define the files to process
    files_to_process = [
        (FILE_1_PATH, FILE_1_OUTPUT_NAME),
        (FILE_2_PATH, FILE_2_OUTPUT_NAME)
    ]
    
    for f_path, f_name in files_to_process:
        process_csv(
            input_path=f_path,
            output_name=f_name,
            text_col=TEXT_COLUMN_NAME,
            n_samples=NUM_SAMPLES,
            test_size=TEST_SPLIT_SIZE,
            seed=RANDOM_SEED
        )
    
    print("All files processed.")