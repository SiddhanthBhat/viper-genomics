import numpy as np

MAP = {
    "A": [1,0,0,0],
    "C": [0,1,0,0],
    "G": [0,0,1,0],
    "T": [0,0,0,1],
    "N": [0,0,0,0]
}

def one_hot_encode(seq):
    return np.array([MAP.get(b, MAP["N"]) for b in seq])
