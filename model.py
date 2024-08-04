from transformers import AutoModel
import torch

class CodeBERTModel(torch.nn.Module):
    def __init__(self):
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.fc = torch.nn.Linear(768, 2)
        
    def forward(self, input):
        out = self.encoder(input)[0]
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out