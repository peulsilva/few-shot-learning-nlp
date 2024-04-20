import numpy as np
from datasets import Dataset
import pandas as pd

def stratified_train_test_split(
    dataset : Dataset,
    classes : np.ndarray,
    train_size : float
):
    """
    Split the dataset into training and validation sets while preserving the class distribution.

    Args:
        dataset (Union[pd.DataFrame, datasets.Dataset]): The input dataset. It can be either a pandas DataFrame or a HuggingFace Dataset.
        classes (np.ndarray): The array of unique class labels present in the dataset.
        train_size (float): The proportion of the dataset to include in the training split. Should be a float in the range (0, 1) if expressed as a fraction, or an integer if expressed as a number of samples.

    Returns:
        Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]: A tuple containing two dictionaries representing the training and validation data splits.
            Each dictionary contains two keys: 'label' and 'text'.
            - The 'label' key corresponds to a list of class labels.
            - The 'text' key corresponds to a list of text samples.

    Notes:
        - Ensure that the dataset contains columns named 'label' and 'text' representing the class labels and text samples, respectively.
        - The 'label' column should contain categorical class labels.
        - The 'text' column should contain textual data.
        - If the dataset is a pandas DataFrame, it should be in the format where each row represents a sample, and each column represents a feature.
        - If the dataset is a cuDF DataFrame, it should be in a similar format as a pandas DataFrame.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import fetch_20newsgroups
        >>> newsgroups_data = fetch_20newsgroups(subset='all')
        >>> df = pd.DataFrame({'text': newsgroups_data.data, 'label': newsgroups_data.target})
        >>> classes = np.unique(df['label'].values)
        >>> train_data, validation_data = stratified_train_test_split(df, classes, train_size=0.8)
    """
    if isinstance(dataset, pd.DataFrame):
        df = dataset

    else:
        df = dataset.to_pandas()
        
    indexes_dict = {}
    for label in classes:
        indexes_dict[label] = df.query(f"label == {label}")


    
    train_data = {
        'label': [],
        'text': []
    }

    validation_data = {
        "label" : [],
        "text": []
    }

    class_proportion = df['label'].value_counts()/len(df)

    # generating train data
    for label in classes:
        n = len(indexes_dict[label])

        if isinstance(train_size, int):
            size = int(class_proportion[label] * train_size)
            
        else:
            size = int(train_size * n)

        train_data['text'] += df.query(f"label == {label}")\
            [0:size]\
            ['text']\
            .to_list()
        
        train_data['label'] += [label]*size
        
        validation_data['text'] +=df.query(f"label == {label}")\
            [size:]\
            ['text']\
            .to_list()
        
        validation_data['label'] += [label]* (n-size)

    return train_data, validation_data