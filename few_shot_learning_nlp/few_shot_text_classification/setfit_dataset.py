import torch 
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import InputExample
from typing import List, Dict
import numpy as np

class SetFitDataset(Dataset):
    def __init__(
        self, 
        text : List[str],
        labels : List[int],
        R : int = -1,
        input_example_format : bool = True
    ) -> None:
        """
        Dataset class for creating pairs of texts with their corresponding labels for training.

        Args:
            text (List[str]): List of texts.
            labels (List[int]): List of corresponding labels.
            R (int, optional): Radius for data expansion. If negative, considers all possible pairs within the dataset. If positive, randomly selects pairs within the specified radius. Defaults to -1.
            input_example_format (bool, optional): If True, returns expanded data in the InputExample format. If False, returns expanded data as a list of lists. Defaults to True.

        Attributes:
            data (List[Union[InputExample, List[str, str, int]]]): Expanded dataset containing pairs of texts with their labels.

        Methods:
            expand_data(X, y, R, input_example_format): Static method to expand the dataset by creating pairs of texts with their corresponding labels.
            __len__(): Returns the length of the dataset.
            __getitem__(index): Returns the item at the specified index in the dataset.
        """
        
        self.data = SetFitDataset.expand_data(text, labels, R, input_example_format)

    @staticmethod
    def expand_data(
        X : List[str],
        y : List[int], 
        R : int = -1,
        input_example_format : bool = True
    ):
        """
        Expand the dataset by creating pairs of texts with their corresponding labels.

        Args:
            X (List[str]): List of texts.
            y (List[int]): List of corresponding labels.
            R (int, optional): Radius for data expansion. If negative, considers all possible pairs within the dataset. If positive, randomly selects pairs within the specified radius. Defaults to -1.
            input_example_format (bool, optional): If True, returns expanded data in the InputExample format. If False, returns expanded data as a list of lists. Defaults to True.

        Returns:
            List[Union[InputExample, List[str, str, int]]]: Expanded dataset containing pairs of texts with their labels.
        """
        expanded_data = []
        for i in range(len(X)):
            if R < 0:
                upper_bound = len(X)
            
            else:
                upper_bound = R
            for j in range(i+1, min(i+1+upper_bound, len(X))):
                label_i = y[i]
                if R > 0:
                    idx_j = np.random.randint(i+1, len(X))

                else: 
                    idx_j = j
                label_j = y[idx_j]

                y_ = int(label_i == label_j)

                if input_example_format:
                
                    expanded_data.append(
                        InputExample(texts = [X[i], X[idx_j]], label =  float(y_) )
                    )
                
                else:
                    expanded_data.append(
                        [X[i], X[idx_j], y_] 
                    )

        return expanded_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]