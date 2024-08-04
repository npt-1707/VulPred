from dataset import VulFixDataset
from torch.utils.data import ConcatDataset, DataLoader
from transformers import BertModel
from argparse import ArgumentParser
import torch
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

def do_train(model, optimizer, criterion, train_loader, valid_loader, epochs, lr):
    model.train()
    print("Start training")
    max_f1 = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        n_samples = 0
        for idx, batch in tqdm(enumerate(train_loader)):
            _, infos, labels = batch
            infos, labels = infos.to(device), labels.to(device)
            outputs = model(infos)
            loss = criterion(outputs, labels)
            total_loss += outputs.shape[0]
            n_samples += len(labels)
            if idx % 10 == 0:
                print(f"Loss: {total_loss/n_samples}")
            loss.backward()
            optimizer.step()
        
        print(f"Loss: {total_loss/len(train_loader)}")
        precision, recall, f1 = do_test(valid_loader, model)
        if f1 > max_f1:
            max_f1 = f1
            if not os.path.exists("model"):
                os.makedirs("model")
            torch.save(model.state_dict(), "model/model.pth")
            print("Model saved")
    print("Finished training")
            
def do_test(dataloader, model):
    model.eval()
    cfx_matrix = np.array([[0, 0], [0, 0]])
    for idx, batch in tqdm(enumerate(dataloader)):
        _, infos, labels = batch
        infos, labels = infos.to(device), labels.to(device)
        outputs = model(infos)

        output = F.softmax(outputs)
        output = outputs.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()

        cfx_matrix += confusion_matrix(label, pred, labels = [0, 1])

    (tn, fp), (fn, tp) = cfx_matrix
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("[EVAL] Precision {}, Recall {}, F1 {}".format(precision, recall, f1))
    return precision, recall, f1



def load_args():
    parser = ArgumentParser(description="Fix prediction")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--path", type=str, default="~/Downloads/{}_{}.csv")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mode", type=str, default="train")
    return parser.parse_args()

if __name__ == "__main__":
    args = load_args()
    path = args.path
    labels = ["pos", "neg"]
    types = ["train", "valid", "test"]

    datasets = {
        type+label: VulFixDataset(path.format(type, label)) for type in types for label in labels
    }

    for type in types:
        datasets[type] = ConcatDataset([datasets[type+label] for label in labels])

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(
        datasets["valid"], batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=args.batch_size, shuffle=False
    )

    model = BertModel().from_pretrained("microsoft/codebert-base")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.mode == "train":
        do_train(model, optimizer, criterion, train_loader, valid_loader, args.epochs)
        do_test(test_loader, model)
    else:
        if os.path.exists("model"):
            model.load_state_dict(torch.load("model/model.pth"))
        do_test(test_loader, model)