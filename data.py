def get_data_from_github(split, category):
    """
    Downloads a specific data file from the GitHub repo,
    skipping if the file already exists.
    """

    # 1. Define the URL and the local output path
    base_url = "https://raw.githubusercontent.com/latoohey/modular-fudge/refs/heads/main/data/splits/"
    filename = f"{category}.txt"
    url = f"{base_url}{split}/{category}/{filename}"

    out_dir = Path("splits") / split
    out_file_path = out_dir / filename

    # --- ADDED CHECK ---
    # 2. Check if the file already exists before doing anything else
    if out_file_path.is_file():
        print(f"File {out_file_path} already exists. Skipping download.")
        return  # Exit the function early
    # -------------------

    try:
        # 3. Create the directory (if it doesn't exist)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 4. Make the HTTP request
        print(f"Downloading {url}...")
        response = requests.get(url)

        # 5. Check if the request was successful
        response.raise_for_status()

        # 6. Write the content to the file
        out_file_path.write_bytes(response.content)

        print(f"Successfully saved to {out_file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except IOError as e:
        print(f"Error writing file: {e}")

def collate(batch):
    """
    Updated collate function that pads with the correct pad_id
    instead of hardcoding 0.
    """
    # Retrieve the dynamic pad_id passed from the SplitLoader (Index 2)
    pad_id = batch[0][2]

    inputs = [b[0] for b in batch]
    lengths = torch.LongTensor([b[1] for b in batch])
    max_length = lengths.max()

    for i in range(len(inputs)):
        diff = max_length - len(inputs[i])

        if diff > 0:
            # --- FIX ---
            # Use torch.full to create a tensor filled with the specific pad_id
            padding = torch.full((diff,), pad_id, dtype=torch.long)
            inputs[i] = torch.cat([inputs[i], padding], dim=0)

    inputs = torch.stack(inputs, dim=0)

    # Get the single integer label (index 3)
    classification_labels = [b[3] for b in batch]
    classification_labels = torch.LongTensor(classification_labels)

    return (inputs, lengths, classification_labels)

class Dataset:
    def __init__(self, args):
        print('Loading data...')
        self.data_dir = args.data_dir
        self.max_len = getattr(args, 'max_len', 512)

        # 1. Tokenizer Setup
        tokenizer_name = getattr(args, 'tokenizer_name', 'distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 2. Pad Token Logic (Restored exactly)
        if self.tokenizer.pad_token is None:
            if '<|finetune_right_pad_id|>' in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = '<|finetune_right_pad_id|>'
            elif '<|reserved_special_token_0|>' in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = '<|reserved_special_token_0|>'
            else:
                print("Warning: No dedicated pad token found. Using EOS token as PAD.")
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer_pad_id = self.tokenizer.pad_token_id
        print(f"Dataset initialized. Pad Token: {self.tokenizer.pad_token} (ID: {self.tokenizer_pad_id})")

        # 3. Data Loading
        train, val, test = [], [], []

        # Helper to determine split size
        # Tries to find args.val_size, falls back to global VAL_SIZE, or 0
        target_val_size = getattr(args, 'val_size', globals().get('VAL_SIZE', 0))

        # --- LOAD TRAIN & VAL ---
        for category, label in [(args.pos_cat, 1), (args.neg_cat, 0)]:

            # A. Path Logic (Restored)
            if getattr(args, 'on_colab', False):
                file_path = os.path.join('splits', 'train', f'{category}.txt')
            else:
                file_path = os.path.join(self.data_dir, 'splits', 'train', f'{category}.txt')

            # B. Download Logic (Restored)
            if not os.path.exists(file_path):
                if getattr(args, 'on_colab', False):
                    print(f"Downloading train/{category} from GitHub...")
                    # Assumes get_data_from_github is defined in your main script context
                    get_data_from_github('train', category)
                else:
                    print(f"Warning: Train file not found at {file_path}. Skipping.")
                    continue

            # C. Processing
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as rf:
                    # We load all lines first to handle the split correctly
                    # We also apply the MIN_LENGTH filter here to keep Map-Style efficient
                    lines = [line.strip() for line in rf if len(line.strip().split()) >= args.min_sentence_length]

                    # Split Logic: First (val_size // 2) items go to Val
                    # If val_size is 0/None, we default to 10% split
                    limit = target_val_size // 2 if target_val_size > 0 else len(lines) // 10

                    val.extend([(l, label) for l in lines[:limit]])
                    train.extend([(l, label) for l in lines[limit:]])

        # --- LOAD TEST ---
        for category, label in [(args.pos_cat, 1), (args.neg_cat, 0)]:

            # A. Path Logic
            if getattr(args, 'on_colab', False):
                file_path = os.path.join('splits', 'test', f'{category}.txt')
            else:
                file_path = os.path.join(self.data_dir, 'splits', 'test', f'{category}.txt')

            # B. Download Logic
            if not os.path.exists(file_path):
                if getattr(args, 'on_colab', False):
                    print(f"Downloading test/{category} from GitHub...")
                    get_data_from_github('test', category)
                else:
                    print(f"Warning: Test file not found at {file_path}. Skipping.")
                    continue

            # C. Processing
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as rf:
                    lines = [line.strip() for line in rf if len(line.strip().split()) >= args.min_sentence_length]
                    test.extend([(l, label) for l in lines])

        self.splits = {'train': train, 'val': val, 'test': test}
        print(f"Counts - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def shuffle(self, split, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])

    def loader(self, split, batch_size, num_workers=0, mode='fudge'):
        data_source = self.splits[split]

        if mode == 'judge':
            collate_fn = self.collate_for_judge
        else:
            collate_fn = self.collate_for_fudge

        return DataLoader(
            SplitLoader(data_source, self),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

    # --- COLLATORS ---
    def collate_for_fudge(self, batch):
        input_ids_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer_pad_id
        )
        lengths = torch.tensor([len(x) for x in input_ids_list], dtype=torch.long)
        return padded_inputs, lengths, labels

    def collate_for_judge(self, batch):
        input_ids_list = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer_pad_id
        )
        attention_mask = (padded_inputs != self.tokenizer_pad_id).long()
        return {
            'input_ids': padded_inputs,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        
class SplitLoader(TorchDataset):
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_sentence, label = self.data[idx]

        encoded = self.parent.tokenizer(
            raw_sentence,
            truncation=True,
            max_length=self.parent.max_len,
            add_special_tokens=True
        )
        return torch.tensor(encoded['input_ids'], dtype=torch.long), label