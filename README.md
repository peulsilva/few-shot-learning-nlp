# few-shot-learning-nlp

This library provides tools and utilities for Few Shot Learning in Natural Language Processing (NLP).

## Overview

Few Shot Learning in NLP involves training and evaluating models on tasks with limited labeled data. This library offers functionalities to facilitate this process.

## Installation

You can install this library via pip:

```bash
pip install -U few-shot-learning-nlp
```

## Documentation

The documentation for this library is available [here](https://peulsilva.github.io/few-shot-learning-nlp/).

## Supported Approaches

### Text Classification
- Sentence Transformers Finetuning ([SetFit](https://arxiv.org/abs/2209.11055))
- Pattern Exploiting Training ([PET](https://arxiv.org/abs/2001.07676))

### Named Entity Recognition for Image Documents
- Pattern Exploiting Training ([PET](https://arxiv.org/abs/2001.07676))
- [Classbite](https://arxiv.org/abs/2305.04928)
- [ContaiNER](https://arxiv.org/abs/2109.07589)

### Classification Utils
- [Focal Loss function for imbalanced datasets](https://arxiv.org/abs/1708.02002)
- Stratified train test split

## Usage

To utilize this library, import the necessary classes and methods and follow the provided [documentation](https://peulsilva.github.io/few-shot-learning-nlp/) for each component.

Here is a short example of the SetFit implementation


```python
from datasets import load_dataset
import pandas as pd
from few_shot_learning_nlp.utils import stratified_train_test_split
from torch.utils.data import DataLoader
from few_shot_learning_nlp.few_shot_text_classification.setfit_dataset import SetFitDataset

# Load a dataset for text classification
ag_news_dataset = load_dataset("ag_news")

# Extract necessary information from the dataset
num_classes = len(ag_news_dataset['train'].features['label'].names)

# Perform few-shot learning by selecting a limited number of classes
n_shots = 50
train_validation, test_df = stratified_train_test_split(ag_news_dataset['train'], num_shots_per_class=n_shots)
train_df, val_df = stratified_train_test_split(pd.DataFrame(train_validation), num_shots_per_class=30)

# Create SetFitDataset objects for training and validation
set_fit_data_train = SetFitDataset(train_df['text'], train_df['label'], input_example_format=True)
set_fit_data_val = SetFitDataset(val_df['text'], val_df['label'], input_example_format=False)

# Create DataLoader objects for training and validation datasets
train_dataloader = DataLoader(set_fit_data_train.data, shuffle=False)
val_dataloader = DataLoader(set_fit_data_val)
```

### Defining Classifier

```python
import torch

class CLF(torch.nn.Module):
    def __init__(
        self,
        in_features : int,
        out_features : int, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layer1 = torch.nn.Linear(in_features, 128)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(128, 32)
        self.layer3 = torch.nn.Linear(32, out_features)

    def forward(self, x : torch.Tensor):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        return self.layer3(x)
```

### Training the Embedding Model <a name="training-the-embedding-model"></a>

```python
import torch
from sentence_transformers import SentenceTransformer
from few_shot_learning_nlp.few_shot_text_classification.setfit import SetFitTrainer

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("whaleloops/phrase-bert")

# Initialize the SetFitTrainer with embedding model and classifier
embedding_model = model.to("cuda")
in_features = embedding_model.get_sentence_embedding_dimension()
clf = CLF(in_features, num_classes).to("cuda")
trainer = SetFitTrainer(embedding_model, clf, num_classes)

# Train the embedding model
trainer.train_embedding(train_dataloader, val_dataloader, n_epochs=10)
```

### Training the Classifier Model <a name="training-the-classifier-model"></a>

```python

# Shuffle training data
_, class_counts = np.unique(train_df['label'], return_counts=True)
X_train_shuffled, y_train_shuffled = shuffle_two_lists(train_df['text'], train_df['label'])

# Train the classifier
history, embedding_model, clf = trainer.train_classifier(
    X_train_shuffled, y_train_shuffled, val_df['text'], val_df['label'],
    clf=CLF(in_features, num_classes),
    n_epochs=15,
    lr=1e-4
)
```

### Testing the Models <a name="testing-the-models"></a>

```python
y_true, y_pred = trainer.test(test_df)
```


