import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred):
    # Threshold predictions
    y_pred_bin = (y_pred > 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred_bin)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred_bin, average="binary", zero_division=0)
    return acc, p, r, f