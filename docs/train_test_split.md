# Stratified Train-Test Split

This function splits a dataset into training and validation sets while preserving the class distribution.

## Usage

To use this function, follow the example below:

```python
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Load dataset
newsgroups_data = fetch_20newsgroups(subset='all')
df = pd.DataFrame({'text': newsgroups_data.data, 'label': newsgroups_data.target})
classes = np.unique(df['label'].values)

# Split dataset
train_data, validation_data = stratified_train_test_split(df, classes, train_size=0.8)
```

### Parameters

- `dataset` (Union[pd.DataFrame, datasets.Dataset]): The input dataset.
- `classes` (np.ndarray): The array of unique class labels present in the dataset.
- `train_size` (Union[float, int]): The proportion of the dataset to include in the training split. Should be a float in the range (0, 1) if expressed as a fraction, or an integer if expressed as a number of samples.

### Returns

A tuple containing two dictionaries representing the training and validation data splits:

- Each dictionary contains two keys: 'label' and 'text'.
- The 'label' key corresponds to a list of class labels.
- The 'text' key corresponds to a list of text samples.

### Notes

- Ensure that the dataset contains columns named 'label' and 'text' representing the class labels and text samples, respectively.
- The 'label' column should contain categorical class labels.
- The 'text' column should contain textual data.
- If the dataset is a pandas DataFrame, it should be in the format where each row represents a sample, and each column represents a feature.

```python
import numpy as np
from datasets import Dataset
import pandas as pd

train_data, validation_data = stratified_train_test_split(df, classes, train_size=0.8)
```
```