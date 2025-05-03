from collections import Counter
import torch

def compute_class_weights(dataset):
    targets = []
    for _, label in dataset:
        targets.append(int(label)) 
    
    counter = Counter(targets)
    count_0 = counter[0]
    count_1 = counter[1]

    print(f"Class 0 count: {count_0}")
    print(f"Class 1 count: {count_1}")

    pos_weight = torch.tensor([count_0 / count_1], dtype=torch.float)
    return pos_weight
