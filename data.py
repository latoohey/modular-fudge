import os
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from collections import defaultdict

from constants import *

def collate(batch):
    """
    This collate function is unchanged from the original,
    as it's needed to pad the inputs and handle the labels.
    """
    pad_id = batch[0][2] # Now index 2
    inputs = [b[0] for b in batch]
    lengths = torch.LongTensor([b[1] for b in batch])
    max_length = lengths.max()

    for i in range(len(inputs)):
        if len(inputs[i]) < max_length:
            # Pad with 0, as that's the embedding padding_idx
            inputs[i] = torch.cat([inputs[i], torch.zeros(max_length - len(inputs[i])).long()], dim=0)
    
    inputs = torch.stack(inputs, dim=0)
    
    # Get the single integer label (index 3)
    classification_labels = [b[3] for b in batch] 
    classification_labels = torch.LongTensor(classification_labels)
    
    return (inputs, lengths, classification_labels)


class Dataset:
    def __init__(self, args):
        print('Loading data...')
        self.data_dir = args.data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
            
        self.tokenizer_pad_id = self.tokenizer.pad_token_id
        
        train, val, test = [], [], []
    
        # --- Process Train & Val from the 'train' directory ---
        for category, label in [(args.pos_cat, 1), (args.neg_cat, 0)]:
            train_file_path = os.path.join(args.data_dir, 'splits', 'train', f'{category}.txt')
            
            # Check if file exists before opening
            if not os.path.exists(train_file_path):
                print(f"Warning: Train file not found at {train_file_path}. Skipping.")
                continue
                
            with open(train_file_path, 'r', encoding='utf-8') as rf:
                for i, line in enumerate(rf):
                    # ... (line truncation logic) ...
                    
                    if i < VAL_SIZE // 2:
                        val.append((line.strip(), label))
                    else:
                        train.append((line.strip(), label))

        # --- Process Test from the 'test' directory ---
        for category, label in [(args.pos_cat, 1), (args.neg_cat, 0)]:
            # Notice the change from 'train' to 'test' in the path
            test_file_path = os.path.join(args.data_dir, 'splits', 'test', f'{category}.txt')
            
            # Check if file exists before opening
            if not os.path.exists(test_file_path):
                print(f"Warning: Test file not found at {test_file_path}. Skipping.")
                continue
                
            with open(test_file_path, 'r', encoding='utf-8') as rf:
                for line in rf:
                    # ... (line truncation logic) ...
                    test.append((line.strip(), label))
        
        # This part remains the same
        self.splits = {'train': train, 'val': val, 'test': test}
        print('Done loading data. Split sizes:')
        for key in self.splits:
            print(f"{key}: {len(self.splits[key])}")

    def shuffle(self, split, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])

    def loader(self, split, batch_size, num_workers=0, indices=None):
        data = self.splits[split] if indices is None else [self.splits[split][i] for i in indices]
        return DataLoader(
            SplitLoader(data, self), 
            batch_size=batch_size, 
            pin_memory=True, 
            collate_fn=collate, 
            num_workers=num_workers
        )


class SplitLoader(IterableDataset):
    def __init__(self, data, parent):
        super().__init__()
        self.data = data
        self.pos = 0
        self.parent = parent

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.pos = 0 # Reset for new epoch
        return self
    
    def __next__(self):
        # This logic is simplified from the original multi-threaded worker logic
        # for clarity. It will work correctly with num_workers=0.
        if self.pos >= len(self):
            raise StopIteration
            
        raw_sentence, classification_label = self.data[self.pos]
        self.pos += 1

        sentence_tokens = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
        length = len(sentence_tokens)

        if length < MIN_SENTENCE_LENGTH:
            # Skip this item and try the next one
            return self.__next__()

        pad_id = self.parent.tokenizer_pad_id
        
        # Return (input_tokens, length, pad_id, label)
        # collate fn will handle the pad_id
        return (sentence_tokens, length, pad_id, classification_label)