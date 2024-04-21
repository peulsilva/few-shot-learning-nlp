# FocalLoss 


## Introduction
The `FocalLoss` class implements the [Focal Loss](https://arxiv.org/abs/1708.02002) criterion, which is used to address class imbalance in classification tasks. It is a modification of the standard cross-entropy loss that down-weights well-classified examples and focuses on hard-to-classify examples.

## Usage example

```python
import torch
import numpy as np

from few_shot_learning_nlp.loss import FocalLoss

# Assuming train_df is your training DataFrame
_, class_counts = np.unique(train_df['label'], return_counts=True)

# Calculate alpha
alpha = len(train_df['label']) / class_counts

# Initialize FocalLoss
loss_fn = FocalLoss(alpha, gamma=2)
```

## Implementation

### Initialization
```python
def __init__(
    self, 
    alpha: Union[list, float, int], 
    gamma: int = 2,
    device: str = 'cuda'
):
```
- `alpha`: The weight factor(s) for each class to address class imbalance. If a single value is provided, it is assumed to be the weight for the positive class, and the weight for the negative class is calculated as 1 - alpha. If a list is provided, it should contain weight factors for each class.
- `gamma`: The focusing parameter to control the degree of adjustment for misclassified samples. Higher values of gamma give more weight to hard-to-classify examples, reducing the influence of easy examples (default: 2).
- `device`: The device on which to perform calculations ('cuda' or 'cpu') (default: 'cuda').

### Attributes
- `alpha` (torch.Tensor): The calculated weight factors for each class.
- `gamma` (float): The focusing parameter.
- `device` (str): The device on which calculations are performed.

### Methods

1. `forward(inputs, targets)`
    - Compute the Focal Loss given the input predictions and target labels.
    - Args:
        - `inputs` (torch.Tensor): The input predictions or logits from the model.
        - `targets` (torch.Tensor): The target class labels.
    - Returns:
        - `torch.Tensor`: The computed Focal Loss value.

### Note
- Ensure that the inputs are logits or unnormalized probabilities, and the targets are class labels.
- This implementation supports binary or multiclass classification tasks.

