
# ImageLayoutDataset

A PyTorch Dataset for handling image layout data with tokenization and labeling.

## Arguments

- `data (List[Dict])`: A list of dictionaries containing the image layout data. Each dictionary should contain at least the following keys:
    - `'words'`: List of words in the text.
    - `'bboxes'`: List of bounding boxes corresponding to each word.
    - `'ner_tags'`: List of named entity recognition tags.
- `tokenizer`: The tokenizer to tokenize the text.
- `device (str, optional)`: The device where tensors will be placed. Defaults to 'cuda'.
- `encode (bool, optional)`: Whether to encode the data during initialization. Defaults to True.
- `tokenize_all_labels (bool, optional)`: Whether to tokenize all labels or only the first token of a word. Defaults to False.
- `valid_labels_keymap (Dict, optional)`: A dictionary mapping valid labels to their corresponding token ids. Defaults to None.

## Methods

1. `tokenize_labels(ner_tags, tokens)`: Tokenizes and aligns the labels with the tokens.
2. `tokenize_boxes(words, boxes)`: Tokenizes the bounding boxes and pads them to match the sequence length.
3. `encode(example)`: Encodes an example from the dataset.
4. `__getitem__(index)`: Retrieves an item from the dataset at the specified index.
5. `__len__()` : Returns the length of the dataset.

## Attributes

- `tokenizer`: The tokenizer used for tokenization.
- `device (str)`: The device where tensors will be placed.
- `valid_labels_keymap (Dict)`: A dictionary mapping valid labels to their corresponding token ids.
- `tokenize_all_labels (bool)`: Whether to tokenize all labels or only the first token of a word.
- `X (List)`: List to store the encoded data or raw data.

## Usage Example

```python
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from few_shot_learning_nlp.few_shot_ner_image_documents.image_dataset import ImageLayoutDataset

# Load the FUNSD dataset
dataset = load_dataset("nielsr/funsd")

# Example tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example data from the FUNSD dataset
data = dataset["train"]

# Initialize the dataset
image_layout_dataset = ImageLayoutDataset(data, tokenizer)

# Get the length of the dataset
print("Dataset length:", len(image_layout_dataset))

# Get an item from the dataset
example_item = image_layout_dataset[0]
print("Example item:", example_item)

# DataLoader example
loader = DataLoader(image_layout_dataset, batch_size=4, shuffle=True)
for batch in loader:
    print("Batch shape:", batch.shape)

```