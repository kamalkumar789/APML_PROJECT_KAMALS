from collections import Counter
import torch

def compute_class_weights(dataset):
    targets = []

    # Collect all labels
    for _, _, label in dataset:
        try:
            targets.append(int(label))
        except:
            print(f"{_}")
            raise ValueError(f"Invalid label value: {label}")

    # Count label frequencies
    counter = Counter(targets)
    count_0 = counter.get(0, 0)
    count_1 = counter.get(1, 0)


    # Safety check
    if count_1 == 0:
        raise ValueError("Class 1 has zero samples; cannot compute pos_weight.")
    if count_0 == 0:
        raise ValueError("Class 0 has zero samples; cannot compute pos_weight.")

    # Compute positive weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([count_0 / count_1], dtype=torch.float)

    return pos_weight
