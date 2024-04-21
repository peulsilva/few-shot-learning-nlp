from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict
import torch
from copy import copy


class ImageLayoutDataset(Dataset):
    """
    A PyTorch Dataset for handling image layout data with tokenization and labeling.

    Args:
        data (List[Dict]): A list of dictionaries containing the image layout data.
        tokenizer: The tokenizer to tokenize the text.
        device (str, optional): The device where tensors will be placed. Defaults to 'cuda'.
        encode (bool, optional): Whether to encode the data during initialization. Defaults to True.
        tokenize_all_labels (bool, optional): Whether to tokenize all labels or only the first token of a word. Defaults to False.
        valid_labels_keymap (Dict, optional): A dictionary mapping valid labels to their corresponding token ids. Defaults to None.

    Methods:
        tokenize_labels(ner_tags, tokens):
            Tokenizes and aligns the labels with the tokens.
        tokenize_boxes(words, boxes):
            Tokenizes the bounding boxes and pads them to match the sequence length.
        encode(example):
            Encodes an example from the dataset.
        __getitem__(index):
            Retrieves an item from the dataset at the specified index.
        __len__():
            Returns the length of the dataset.

    Attributes:
        tokenizer: The tokenizer used for tokenization.
        device (str): The device where tensors will be placed.
        valid_labels_keymap (Dict): A dictionary mapping valid labels to their corresponding token ids.
        tokenize_all_labels (bool): Whether to tokenize all labels or only the first token of a word.
        X (List): List to store the encoded data or raw data.
    """
    def __init__(self, 
                 data,
                 tokenizer,
                 device : str = 'cuda',
                 encode : bool = True,
                 tokenize_all_labels : bool = False,
                 valid_labels_keymap : Dict = None) -> None:
        """
        Initializes the ImageLayoutDataset with the provided parameters.

        Args:
            data (List[Dict]): A list of dictionaries containing the image layout data.
                            Each dictionary should contain at least the following keys:
                                - 'words': List of words in the text.
                                - 'bboxes': List of bounding boxes corresponding to each word.
                                - 'ner_tags': List of named entity recognition tags.
            tokenizer: The tokenizer to tokenize the text.
            device (str, optional): The device where tensors will be placed. Defaults to 'cuda'.
            encode (bool, optional): Whether to encode the data during initialization. Defaults to True.
            tokenize_all_labels (bool, optional): Whether to tokenize all labels or only the first token of a word. Defaults to False.
            valid_labels_keymap (Dict, optional): A dictionary mapping valid labels to their corresponding token ids. Defaults to None.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device
        self.valid_labels_keymap = valid_labels_keymap
        self.tokenize_all_labels = tokenize_all_labels

        if encode:
            self.X = []
            for example in tqdm(data):
                if 'words' not in example or 'bboxes' not in example or 'ner_tags' not in example:
                    raise ValueError("Each example dictionary in the 'data' list must contain 'words', 'bboxes', and 'ner_tags' keys.")
                X= self.encode(example)
                self.X.append(X)

        else:
            self.X = data
    
    def tokenize_labels(
        self,
        ner_tags : List,
        tokens 
    )-> torch.Tensor:
        """
        Tokenizes and aligns the labels with the tokens.

        Args:
            ner_tags (List): The labels.
            tokens (_type_): The tokens.

        Returns:
            torch.Tensor: The tokenized labels.
        """
        labels = []

        word_ids = tokens.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                if self.valid_labels_keymap is not None:
                    label_ids.append(self.valid_labels_keymap[ner_tags[word_idx]])
                else:
                    label_ids.append(ner_tags[word_idx])
            else:
                if self.tokenize_all_labels:
                    if self.valid_labels_keymap is not None:
                        label_ids.append(self.valid_labels_keymap[ner_tags[word_idx]])
                    else:
                        label_ids.append(ner_tags[word_idx])
                else: 
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

        return torch.Tensor(labels).to(torch.int64)
    
    
    def tokenize_boxes(
        self,
        words : List,
        boxes : List,
    ):
        """
        Tokenizes the bounding boxes and pads them to match the sequence length.

        Args:
            words (List): The list of words.
            boxes (List): The list of bounding boxes.

        Returns:
            torch.Tensor: The tokenized bounding boxes.
        """
        token_boxes = []
        max_seq_length = 512
        pad_token_box = [0,0,0,0]
        
        for word, box in zip(words, boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        # Truncation of token_boxes
        special_tokens_count = 2 
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        # add bounding boxes of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length

        return torch.tensor(token_boxes)

    def encode(
        self,
        example, 
    ):
        """
        Encodes an example from the dataset.

        Args:
            example: The example data.

        Returns:
            Dict[str, torch.Tensor]: The encoded tokens.
        """
        words = example['words']
        boxes = example['bboxes']
        # image = Image.open(example['image_path'])s
        word_labels = example['ner_tags']

        
        tokens = self.tokenizer(
            words, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
            is_split_into_words= True
        )

        labels = self.tokenize_labels(word_labels,tokens)
        bbox = self.tokenize_boxes(words, boxes)

        tokens = {
            **tokens,
            "labels": labels,
            "bbox": bbox
        }

        for (k,v ) in tokens.items():
            tokens[k] = v.to(self.device)
    
        return tokens

        
    
    def __getitem__(self, index: int):
        return self.X[index]

    def __len__(self):
        return len(self.X)