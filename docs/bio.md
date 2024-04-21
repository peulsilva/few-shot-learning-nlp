
# Bio Technique for Named Entity Recognition on Image Documents

## Introduction

The [BioTechnique](https://arxiv.org/abs/2305.04928) approach adapts multi-class classification into a binary framework by predicting the class membership of each token in a text. It achieves this by appending each token with its possible class label, thereby increasing the data volume by a factor of k, where k represents the number of classes. This augmentation enables binary classification for each token, facilitating more granular classification within the document.

In the BioTechnique approach, suppose we have the classes C = [City, Transport, None]. For a given sentence "Paris has a good metro system", we transform it into three separate sentences:

1. City [SEP] Paris has a good metro system

2. Transport [SEP] Paris has a good metro system

3. None [SEP] Paris has a good metro system

Each sentence is treated as a binary classification task to predict whether each token belongs to its corresponding class or not. This approach effectively increases the data volume and allows for more precise classification within the document.

## Usage example

### Imports

```python
%load_ext autoreload
%autoreload 2

import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score, multiclass_confusion_matrix, binary_f1_score

from few_shot_learning_nlp.few_shot_ner_image_documents.bio_technique import BioTrainer
from few_shot_learning_nlp.few_shot_ner_image_documents.image_dataset import ImageLayoutDataset
from few_shot_learning_nlp.few_shot_ner_image_documents.bio_technique_dataset import generate_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Pre-processing

The dataset is loaded from the FUNSD dataset, and necessary pre-processing steps are performed, including tokenization and dataset generation.

### Training

```python
# Loading FUNSD Dataset
funsd_dataset = load_dataset("nielsr/funsd")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Generating dataset
train_data = generate_dataset(funsd_dataset['train'], label_names, idx2label, tokenizer, n_shots=2)
val_data = generate_dataset(Dataset.from_dict(funsd_dataset['train'][10:]), label_names, idx2label, tokenizer, n_shots=50)
test_data = generate_dataset(funsd_dataset['test'], label_names, idx2label, tokenizer, n_shots=np.inf)

# Creating DataLoader
train_dataset = ImageLayoutDataset(train_data, tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=False)

validation_dataset = ImageLayoutDataset(val_data, tokenizer)
validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=4)

test_dataset = ImageLayoutDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4)

# Initializing and training the model
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

trainer = BioTrainer(model, optimizer, n_classes)

history = trainer.train(train_dataloader, validation_dataloader, n_epochs=100)

# Plotting validation performance
best_f1, best_epoch = np.max(history), np.argmax(history)
plt.plot(history)
plt.scatter([best_epoch], [best_f1], color="green", label=f"Best f1 : {round(best_f1,3)}")
plt.legend()
plt.ylabel("f1 score")
plt.xlabel("epoch")
plt.title("Validation performance - FUNSD - 2 shots")
plt.grid()

```

### Evaluation

The model's performance is evaluated using the test dataset, and the F1 score and confusion matrix are computed.

```python

# Evaluating on the test set
y_true, y_pred = trainer.test(test_dataloader)
f1 = multiclass_f1_score(y_pred.to(torch.int64), y_true.to(torch.int64), num_classes=n_classes)
```

This notebook showcases the implementation of Few-Shot Learning for NER on image documents using transformers and PyTorch.

---
title: BioTrainer Class Documentation
---

## BioTrainer

### Methods

1. `__init__(model, optimizer, n_classes, device="cuda")`

    Initialize BioTrainer with the provided token classification model, optimizer, and other parameters.

    - `model (AutoModelForTokenClassification)`: The token classification model to be trained.
    - `optimizer (torch.optim)`: The optimizer used for training.
    - `n_classes (int)`: The number of classes for token classification.
    - `device (str, optional)`: The device where the model will be trained. Defaults to "cuda".

2. `train(train_dataloader, validation_dataloader, n_epochs=20)`

    Train and validate the token classification model.

    - `train_dataloader (Dataset)`: DataLoader containing the training data. Requires batch size of 1.
    - `validation_dataloader (Dataset)`: DataLoader containing the validation data. Requires batch size of 1.
    - `n_epochs (int, optional)`: Number of epochs for training. Defaults to 20.

    Returns:
    - `history (list)`: History of evaluation metric (F1-score) during training.

3. `test(test_dataloader)`

    Performs testing on the provided test dataloader.

    - `test_dataloader (DataLoader)`: The dataloader containing the test dataset. Requires batch size of 1.

    Returns:
    - `Tuple[torch.Tensor, torch.Tensor]`: A tuple containing the true labels and predicted labels.

### Attributes

- `history (list)`: List to store the evaluation metric (F1-score) history during training.
- `best_model (AutoModelForTokenClassification)`: The best-performing model based on validation F1-score.
- `n_classes (int)`: The number of classes for token classification.
- `model (AutoModelForTokenClassification)`: The token classification model.
- `optimizer (torch.optim)`: The optimizer used for training.
- `device (str)`: The device where the model will be trained.

## generate_dataset Function

Generates a new dataset by modifying the original dataset based on the given parameters.

### Arguments

- `dataset (Dataset)`: The original dataset.
- `label_names (List[str])`: A list of label names to generate the dataset.
- `idx2label (Dict[int, str])`: A dictionary mapping label indices to label names.
- `tokenizer (AutoTokenizer)`: The tokenizer used to tokenize words.
- `n_shots (int)`: The number of shots to consider from the original dataset.

### Returns

- `Dataset`: The generated dataset.

### Notes

- The function generates a new dataset by modifying the original dataset. It creates additional samples based on the provided label names and the number of shots specified.
- Each document in the original dataset is processed to generate new samples. Only a limited number of shots (`n_shots`) are considered from the original dataset.
- For each label in `label_names`, the function creates new samples where the tokens belonging to that label are marked with 1 and other tokens with -100 in the `ner_tags` field.
- The `words` field in the new dataset contains the tokenized words, and the `bboxes` field contains the corresponding bounding boxes.