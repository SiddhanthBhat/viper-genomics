import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.viper import VIPER
from utils import compute_metrics, set_seed
import os
from tqdm import tqdm

def load_data(split_path, batch_size):
    X, y = torch.load(split_path)
    # Ensure y is [Batch, 1]
    if len(y.shape) == 1:
        y = y.unsqueeze(1)
    dataset = TensorDataset(X, y)
    shuffle = "train" in split_path
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def main():
    # Load Config
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    data_dir = cfg['data']['processed_dir']
    train_dl = load_data(os.path.join(data_dir, "train.pt"), cfg['train']['batch_size'])
    val_dl = load_data(os.path.join(data_dir, "val.pt"), cfg['train']['batch_size'])

    # Initialize Model
    model = VIPER(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    criterion = nn.BCELoss()

    best_f1 = 0.0
    
    print("Starting Training...")
    for epoch in range(cfg['train']['epochs']):
        # --- Training Loop ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_dl, desc=f"Epoch {epoch+1} [Train]", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dl)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_dl, desc=f"Epoch {epoch+1} [Val]", leave=False):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                
                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())
        
        avg_val_loss = val_loss / len(val_dl)
        
        # Compute Metrics
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"           | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Checkpoint
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "weights/viper_best.pth")
            print(">>> New Best Model Saved!")

if __name__ == "__main__":
    main()