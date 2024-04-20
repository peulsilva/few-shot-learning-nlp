from sentence_transformers import SentenceTransformer, losses
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score, multiclass_confusion_matrix, multiclass_f1_score
from copy import deepcopy
from typing import List, Tuple
from IPython.display import clear_output
import pandas as pd

class SetFitTrainer():
    def __init__(
        self,
        embedding_model ,
        classifier_model : torch.nn.Module,
        num_classes : int,
        dataset_name : str = None,
        model_name : str = None,
        device : str ='cuda' ,
    ) -> None:
        self.embedding_model = embedding_model.to(device)
        self.device = device
        self.dataset_name = dataset_name
        self.classifier_model = classifier_model.to(device)
        self.num_classes = num_classes
        self.model_name= model_name

    def train_embedding(
            self,
            train_dataloader : DataLoader,
            val_dataloader : DataLoader,
            n_epochs : int = 10,
            filepath : str = None,
            **kwargs
        ):
        """
        Train the embedding model using the provided training dataloader and validate it 
        using the validation dataloader.

        Args:
            train_dataloader (DataLoader): DataLoader containing the training data.
            val_dataloader (DataLoader): DataLoader containing the validation data.
            n_epochs (int, optional): Number of epochs for training. Defaults to 10.
            filepath (str, optional): Filepath to save the best model. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the embedding model's fit method.

        Returns:
            None

        Note:
            This method updates the best model if a higher F1 score is achieved during validation
            and optionally saves the best model to the specified filepath.
        """
        loss_fn = losses.CosineSimilarityLoss(self.embedding_model)
        cos_sim = torch.nn.CosineSimilarity(dim = 1)

        best_f1 = 0
        self.best_model : SentenceTransformer = None

        for epoch in range(n_epochs):
            self.embedding_model.fit(
                train_objectives=[ (train_dataloader, loss_fn)],
                epochs = 1,
                **kwargs
            )

            y_true_val = torch.tensor([],device=self.device)
            y_pred_val = torch.tensor([],device=self.device)

            print(f"Running validation after {epoch} epochs")

            for [x1, x2, y] in tqdm(val_dataloader):
                with torch.no_grad():
                    v1 = self.embedding_model.encode(x1, convert_to_tensor= True)
                    v2 = self.embedding_model.encode(x2, convert_to_tensor= True)

                    cos = cos_sim(v1, v2)

                    y_pred = round(cos.item())
                    y_true = y

                    y_pred_val = torch.cat([
                        y_pred_val, 
                        torch.tensor([y_pred]).to(self.device)
                    ])

                    y_true_val = torch.cat([
                        y_true_val, 
                        torch.tensor([y_true]).to(self.device)
                    ])
            
            conf_matrix= multiclass_confusion_matrix(
                y_pred_val.to(torch.int64),
                y_true_val.to(torch.int64),
                num_classes=self.num_classes
            )
            
            f1 = binary_f1_score(
                y_pred_val,
                y_true_val,
            )
            
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = deepcopy(self.embedding_model)


            clear_output()
            print(f'f1 score: {f1.item()}')
            print(conf_matrix)

        if filepath is not None:
            self.best_model.save(filepath)

    def train_classifier(
        self,
        X_train : List[str],
        y_train : List[int],
        X_val : List[str],
        y_val : List[int],
        n_epochs : int =100,
        loss_fn : torch.nn.Module = torch.nn.CrossEntropyLoss(),
        embedding_model = None,
        clf: torch.nn.Module = None,
        lr : float = 1e-5
    ):
        """
        Train the classifier model using the provided training and validation data.

        Args:
            X_train (List[str]): List of training texts.
            y_train (List[int]): List of corresponding training labels.
            X_val (List[str]): List of validation texts.
            y_val (List[int]): List of corresponding validation labels.
            n_epochs (int, optional): Number of epochs for training. Defaults to 100.
            loss_fn (torch.nn.Module, optional): Loss function for training. Defaults to torch.nn.CrossEntropyLoss().
            embedding_model ([type], optional): Pre-trained embedding model to use. If None, uses the best_model. Defaults to None.
            clf (torch.nn.Module, optional): Classifier model to use. If None, uses self.clf. Defaults to None.
            lr (float, optional): Learning rate for optimizer. Defaults to 1e-5.

        Returns:
            Tuple[List[float], SentenceTransformer, torch.nn.Module]: Tuple containing the history of F1 scores during training, the embedding model, and the best classifier model.
        """
        if embedding_model is None:
            self.embedding_model = self.best_model\
                .to(self.device)
            
        else:
            self.embedding_model = embedding_model

        if clf is not None:
            self.clf = clf

        optimizer = torch.optim.Adam(
            self.classifier_model.parameters(),
            lr = lr
        )

        self.best_clf = None
        best_f1 = 0

        self.history = []

        for epoch in (range(n_epochs)):
            for i in tqdm(range(len(X_train))):
                text = X_train[i]
                label = torch.tensor(y_train[i])\
                    .to(self.device)

                with torch.no_grad():
                    embedding = self.\
                        embedding_model\
                        .encode(text, convert_to_tensor=True)\
                        .to(self.device)

                optimizer.zero_grad()
                output = self.classifier_model(embedding)
                loss = loss_fn(output, label)


                loss.backward()
                optimizer.step()

            y_true_val = torch.tensor([],device=self.device)
            y_pred_val = torch.tensor([],device=self.device)

            for i in range(len(X_val)):
                text = X_val[i]
                label = torch.tensor(y_val[i])\
                    .to(self.device)

                with torch.no_grad():
                    embedding = self\
                        .embedding_model\
                        .encode(text, convert_to_tensor=True)

                    y_pred = self.classifier_model(embedding)\
                        .argmax()
                    
                    y_pred_val = torch.cat([
                        y_pred_val, 
                        torch.tensor([y_pred]).to(self.device)
                    ])

                    y_true_val = torch.cat([
                        y_true_val, 
                        torch.tensor([y_val[i]]).to(self.device)
                    ])

            if self.num_classes == 2:
                f1 = binary_f1_score(
                    y_pred_val,
                    y_true_val
                )

            else:
                    
                f1 = multiclass_f1_score(
                    y_pred_val,
                    y_true_val,
                    num_classes=self.num_classes
                )
            
            self.history.append(f1.item())
            if f1 > best_f1:
                best_f1 = f1
                self.best_clf = deepcopy(self.classifier_model)

            conf_matrix= multiclass_confusion_matrix(
                y_pred_val.to(torch.int64),
                y_true_val.to(torch.int64),
                num_classes=self.num_classes
            )

            clear_output()
            print(f"---------Epoch: {epoch}-----------")
            print(f'f1 score: {f1.item()}')
            print(conf_matrix)

        return self.history, self.embedding_model, self.best_clf

    def test(
        self,
        test_df: pd.DataFrame,
        embedding_model: SentenceTransformer = None,
        clf: torch.nn.Module = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Test the performance of the trained models on the provided test dataset.

        Args:
            test_df (pd.DataFrame): The DataFrame containing the test dataset with 'text' and 'label' columns.
            embedding_model (SentenceTransformer, optional): The SentenceTransformer model for text embedding. 
                If None, the best trained embedding model will be used. Defaults to None.
            clf (torch.nn.Module, optional): The trained classifier model. 
                If None, the best trained classifier will be used. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the true labels and predicted labels for the test dataset.

        Notes:
            - Ensure that the test dataset DataFrame contains 'text' and 'label' columns representing the text samples and their corresponding labels.
            - If `embedding_model` is None, the method will use the best trained embedding model stored in the class attribute `best_model`.
            - If `clf` is None, the method will use the best trained classifier stored in the class attribute `best_clf`.
            - The method iterates through the test dataset, computes embeddings for each text sample, and predicts the labels using the classifier.
            - The predicted labels and true labels are returned as torch tensors.
        """
        if embedding_model is not None:
            self.best_model = embedding_model

        if clf is not None:
            self.best_clf = clf

        y_pred_test = torch.tensor([],device=self.device)
        y_true_test = torch.tensor([],device=self.device)


        for i in tqdm(range(len(test_df['label']))):
            text = test_df['text'][i]

            with torch.no_grad():
                embedding = self.best_model.encode(text, convert_to_tensor=True)

                y_pred = self.best_clf(embedding)\
                    .argmax()
                
                y_pred_test = torch.cat([
                    y_pred_test, 
                    torch.tensor([y_pred]).to( self.device)
                ])

                y_true_test = torch.cat([
                    y_true_test, 
                    torch.tensor([test_df['label'][i]]).to(self.device)
                ])
        
        return y_true_test, y_pred_test