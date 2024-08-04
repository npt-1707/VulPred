from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import pickle
import os
from tqdm import tqdm
import numpy as np

MAX_LENGTH = 512

def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    return data


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


def get_input_and_mask(src, dst, max_length, tokenizer):
    src_tokens = tokenizer.tokenize(src)
    dst_tokens = tokenizer.tokenize(dst)

    tokens = (
        [tokenizer.cls_token]
        + src_tokens
        + [tokenizer.sep_token]
        + dst_tokens
        + [tokenizer.sep_token]
    )

    token_length = len(tokens)
    if token_length > max_length:
        src_length = len(src_tokens)
        dst_length = len(dst_tokens)
        if src_length >= max_length-2:
            new_tokens = [tokenizer.cls_token] + src_tokens[:max_length-2] + [tokenizer.sep_token]
        else:
            new_tokens = [tokenizer.cls_token] + src_tokens + [tokenizer.sep_token] + dst_tokens[:max_length-3-src_length] + [tokenizer.sep_token]
        mask = [1 for i in range(max_length)]
    else:
        new_tokens = [
            tokens[i] if i < token_length else tokenizer.pad_token
            for i in range(max_length)
        ]
        mask = [1 if i < token_length else 0 for i in range(max_length)]

    tokens_ids = np.array(tokenizer.convert_tokens_to_ids(new_tokens))
    mask = np.array(mask)
    if len(tokens_ids) > max_length:
        print(len(dst_tokens))
        print(len(src_tokens))
        print(len(tokens_ids))
        raise "Truncation errors"
    return tokens_ids, mask


class VulFixDataset(Dataset):
    def __init__(self, path, mode="train", type="pos"):
        self.cache = f"cache/{mode}_{type}.pkl"
        if os.path.exists(self.cache):
            self.load_cache()
        else:
            self.data = load_data(path)
            self.process()
            self.save_cache()
            
    def process(self):
        print("Processing data...")
        self.ids = []
        self.infos = []
        self.masks = []
        self.labels = []
        for i in tqdm(range(len(self.data))):
            self.ids.append(self.data.iloc[i]["commit_id"])
            mes = self.data.iloc[i]["commit_message"]
            code = self.data.iloc[i]["diff"]
            info, mask = get_input_and_mask(mes, code, MAX_LENGTH, tokenizer)
            self.infos.append(info)
            self.masks.append(mask)
            self.labels.append(self.data.iloc[i]["label"])
            
    def save_cache(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open(self.cache, "wb") as f:
            pickle.dump([self.ids, self.infos, self.masks, self.labels], f)
        
    def load_cache(self):
        with open(self.cache, "rb") as f:
            self.ids, self.infos, self.masks, self.labels = pickle.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.infos[idx], self.masks[idx], self.labels[idx]
