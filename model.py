from transformers import AutoModel
import torch

class CodeBERTModel(torch.nn.Module):
    def __init__(self):
        super(CodeBERTModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.fc = torch.nn.Linear(768, 2)
        
    def forward(self, input, mask):
        out = self.encoder(input, mask)[0]
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out