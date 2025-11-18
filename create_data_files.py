import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
# 1. Set your input file path (The single CSV with all data)
INPUT_CSV = 'encyclopedia_dataset.csv' 

# 2. Set column names
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'  # <--- The column containing category/label names

# 3. Set split parameters
TEST_SPLIT_SIZE = 0.20  # 80% train, 20% test
RANDOM_SEED = 42

# 4. Output directory setup
BASE_OUTPUT_DIR = 'data'
# ---------------------

def process_dataset(input_path, text_col, label_col, test_size, seed):
    """
    Reads a single CSV, identifies unique labels, splits data per label,
    and organizes them into train/test folders.
    """
    print(f"--- Reading {input_path} ---")
    
    try:
        # Load the dataset
        df = pd.read_csv(input_path)
        
        # Basic validation
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Columns '{text_col}' and '{label_col}' must exist in CSV.")
            
        # Drop rows where text is missing
        df = df.dropna(subset=[text_col])
        
        # Identify unique labels (categories)
        unique_labels = df[label_col].unique()
        print(f"Found {len(unique_labels)} unique labels: {list(unique_labels)}")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Create Base Train/Test Directories
    train_root = os.path.join(BASE_OUTPUT_DIR, 'train')
    test_root = os.path.join(BASE_OUTPUT_DIR, 'test')
    
    # Iterate over each unique label to process them individually
    for label in unique_labels:
        print(f"\nProcessing Label: '{label}'")
        
        # Filter data for this specific label
        label_data = df[df[label_col] == label]
        
        # Check if we have enough data to split
        if len(label_data) < 2:
            print(f"  Warning: Not enough data to split for '{label}'. Skipping.")
            continue

        # Split into train and test
        train_texts, test_texts = train_test_split(
            label_data[text_col],
            test_size=test_size,
            random_state=seed
        )

        # Create label-specific subdirectories
        # Example: dataset_output/train/mystery/
        train_label_dir = os.path.join(train_root, str(label))
        test_label_dir = os.path.join(test_root, str(label))
        
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        # Define output filenames (standardized name inside the folder)
        train_file = os.path.join(train_label_dir, f'{label}.txt')
        test_file = os.path.join(test_label_dir, f'{label}.txt')

        # Helper function to write list to file
        def write_txt(filepath, data_series):
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    for line in data_series:
                        # Convert to string and strip newlines to prevent double spacing
                        clean_line = str(line).replace('\n', ' ').strip()
                        if clean_line:
                            f.write(clean_line + '\n')
                return True
            except IOError as e:
                print(f"  Error writing {filepath}: {e}")
                return False

        # Write the files
        if write_txt(train_file, train_texts):
            print(f"  Saved {len(train_texts)} lines to {train_file}")
            
        if write_txt(test_file, test_texts):
            print(f"  Saved {len(test_texts)} lines to {test_file}")

    print(f"\n--- Processing Complete. Check folder: '{BASE_OUTPUT_DIR}' ---")

# --- Main execution ---
if __name__ == "__main__":
    process_dataset(
        input_path=INPUT_CSV,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN,
        test_size=TEST_SPLIT_SIZE,
        seed=RANDOM_SEED
    )