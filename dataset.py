from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import pickle
import os

MAX_LENGTH = 512

def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    return data


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

class VulFixDataset(Dataset):
    def __init__(self, path, mode="train", type="pos"):
        self.cache = f"cache/{mode}_{type}.pkl"
        if os.path.exists(self.cache):
            self.load_cache()
        else:
            self.data = load_data(path)
            self.save_cache()
            
    def process(self):
        self.ids = []
        self.infos = []
        self.labels = []
        for i in range(self.data):
            self.ids.append(self.data.iloc[i]["commit_id"])
            mes = tokenizer.tokenize(self.data.iloc[i]["message"])
            code = tokenizer.tokenize(self.data.iloc[i]["code"])
            if len(code) > MAX_LENGTH-3-len(mes):
                code = code[:MAX_LENGTH-3-len(mes)]
            info = [tokenizer.cls_token] + mes + [tokenizer.sep_token] + code + [tokenizer.sep_token]
            self.infos.append(info)
            self.labels.append(self.data.iloc[i]["label"])
            
    def save_cache(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open(self.cache, "wb") as f:
            pickle.dump([self.ids, self.infos, self.labels], f)
        
    def load_cache(self):
        with open(self.cache, "rb") as f:
            self.ids, self.infos, self.labels = pickle.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.infos[idx], self.labels[idx]