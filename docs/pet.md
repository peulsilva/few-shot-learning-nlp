# Pattern Exploiting Training (PET)

## Introduction
[Pattern Exploiting Training (PET)](https://arxiv.org/abs/2001.07676) is a technique used for fine-tuning pretrained language models for text classification tasks. It leverages patterns in the input text to improve classification performance. This page provides an overview of how to use PET for text classification, along with a brief usage example.

## Usage example

### Imports
```python
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torcheval.metrics.functional import multiclass_confusion_matrix, binary_f1_score, multiclass_f1_score
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from few_shot_learning_nlp.few_shot_text_classification.pattern_exploiting import PETTrainer
from few_shot_learning_nlp.few_shot_text_classification.pattern_exploiting_dataset import PETDatasetForClassification
from few_shot_learning_nlp.utils import stratified_train_test_split
```

### Preprocessing
```python

# Load AG News dataset
ag_news_dataset = load_dataset("ag_news")

# Define a pattern for PET
def pattern1(text: str, tokenizer: AutoTokenizer):
    return f"{tokenizer.mask_token} news: {text}"

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Get class names and number of classes
class_names = ag_news_dataset['train'].features['label'].names
num_classes = len(class_names)

# Define verbalizer and inverse verbalizer
verbalizer = {idx: tokenizer.vocab[x.lower()] for idx, x in enumerate(class_names)}
inverse_verbalizer = {tokenizer.vocab[x.lower()]: idx for idx, x in enumerate(class_names)}

# Preprocess the data
def preprocess(text: List[str], labels: List[int]):
    processed_text = []
    processed_labels = []
    for idx in range(len(text)):
        label = idx2classes[labels[idx]]
        text_ = text[idx]
        processed_text.append(pattern1(text_, tokenizer))
        processed_labels.append(label)
    return processed_text, processed_labels

train_text, train_labels = preprocess(train_df['text'], train_df['label'])
val_text, val_labels = preprocess(val_df['text'], val_df['label'])
test_text, test_labels = preprocess(test_df['text'], test_df['label'])

# Create datasets and dataloaders
train_dataset = PETDatasetForClassification(train_text, train_labels, tokenizer)
val_dataset = PETDatasetForClassification(val_text, val_labels, tokenizer)
test_dataset = PETDatasetForClassification(test_text, test_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True)
val_dataloader = DataLoader(val_dataset)
test_dataloader = DataLoader(test_dataset)

```

### Training the Models
```python
# Load pretrained model
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)

# Instantiate PETTrainer
trainer = PETTrainer(model, verbalizer, tokenizer, num_classes=len(classes))

# Train the model
history, confusion_matrix, best_model = trainer.train(
    train_dataloader, 
    val_dataloader, 
    alpha=1e-6,
    lr=1e-4,
    device=device
)
```

### Evaluating the Model
```python
# Plot F1 score over epochs
plt.plot(history)
plt.scatter(np.argmax(history), np.max(history), label=f"f1 = {np.max(history)}", c="green")
plt.grid()
plt.title("F1 score over epochs")
plt.xlabel("epoch")
plt.ylabel("F1 score")
plt.legend()

# Plot confusion matrix
df = pd.DataFrame(confusion_matrix.to("cpu").numpy(), index=class_names, columns=class_names)
sns.heatmap(df, annot=True, fmt='2g')
plt.title(f"Confusion Matrix Ag news - PET")

# Test the model
y_true_test, y_pred_test = trainer.test(test_dataloader)
f1 = multiclass_f1_score(y_pred_test, y_true_test, num_classes=len(classes))
```

This code demonstrates how to perform text classification using Pattern Exploiting Training (PET) with a few-shot learning approach. It includes steps for dataset loading, preprocessing, model training, evaluation, and testing.

Make sure to adapt the paths, hyperparameters, and dataset configurations according to your specific setup and requirements.

## PETTrainer 

### Introduction
The `PETTrainer` class is designed to fine-tune a model with Pattern Exploiting Training (PET) for text classification tasks. It combines a pretrained language model with PET to enhance classification performance.

### Initialization
```python
def __init__(
    self,
    model: AutoModelForMaskedLM,
    verbalizer: Dict,
    tokenizer: AutoTokenizer,
    num_classes: int,
    device: str = 'cuda',
) -> None:
```
- `model`: Pretrained language model to be fine-tuned.
- `verbalizer`: Dictionary mapping class labels to corresponding tokens.
- `tokenizer`: Tokenizer associated with the model.
- `num_classes`: Number of classes in the classification task.
- `device`: Device on which calculations are performed (default: 'cuda').

### Methods
1. `get_y_true(input, inverse_verbalizer, device)`
    - Get the true labels from the input data.
    - `input`: Input data containing the true labels.
    - `inverse_verbalizer`: Dictionary mapping tokens to class labels.
    - `device`: Device on which calculations are performed.

2. `train(train_dataloader, val_dataloader, alpha, loss_fn, device, lr, n_epochs)`
    - Train the PET model.
    - `train_dataloader`: DataLoader for the training dataset.
    - `val_dataloader`: DataLoader for the validation dataset.
    - `alpha`: Weighting factor for balancing MLM and CE losses.
    - `loss_fn`: Custom loss function for the CE loss (default: None).
    - `device`: Device on which calculations are performed (default: 'cuda').
    - `lr`: Learning rate for optimization (default: 1e-5).
    - `n_epochs`: Number of training epochs (default: 10).
    - Returns a tuple containing training history, confusion matrix, and the best trained model.

3. `test(test_dataloader)`
    - Perform inference/testing using the trained model on the provided test data.
    - `test_dataloader`: DataLoader containing the test data.
    - Returns a tuple containing true labels and predicted labels.

### Notes
- PET (Pattern Exploiting Training) is a technique for fine-tuning pretrained language models for text classification tasks.
- The trainer supports multi-class classification tasks.
- The model is trained using a combination of Masked Language Model (MLM) loss and Cross-Entropy (CE) loss.


## PETDatasetForClassification

### Introduction
The `PETDatasetForClassification` class is a Dataset class designed for Pattern Exploiting Training (PET) in text classification tasks. It preprocesses input texts, tokenizes them, encodes class labels, and prepares the dataset for PET training.

### Initialization
```python
def __init__(
    self, 
    processed_text: List[str], 
    labels: List[int],
    tokenizer: AutoTokenizer,
    device: str = "cuda"
) -> None:
```
- `processed_text`: List of processed input texts.
- `labels`: List of corresponding class labels.
- `tokenizer`: Tokenizer used to tokenize the input texts.
- `device`: Device on which calculations are performed (default: "cuda").

### Attributes
- `tokens`: Tokenized input texts.
- `encoded_labels`: Encoded labels for PET training.
- `inputs`: Dictionary containing tokenized inputs and encoded labels.
    - Keys: 'input_ids', 'attention_mask', 'labels'
- `device`: Device on which calculations are performed.

### Methods
1. `__getitem__(index)`
    - Retrieves an item from the dataset at the specified index.
    - Returns a dictionary containing tokenized inputs and encoded labels.

2. `__len__()`
    - Returns the total number of items in the dataset.

### Notes
- This dataset class is designed for use with PET (Pattern Exploiting Training) in text classification tasks.
- Each input text is tokenized using the provided tokenizer and padded/truncated to the maximum length.
- The class labels are encoded and replaced with mask tokens for PET training.
- The dataset is prepared for training on the specified device.