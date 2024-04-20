import torch
from typing import Union
import torch.nn.functional as F
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(
        self, 
        alpha : Union[list, float, int], 
        gamma : int =2,
        device : str = 'cuda'
    ):
        """
        Focal Loss criterion for addressing class imbalance in classification tasks.

        Args:
            alpha (Union[list, float, int], optional): The weight factor(s) for each class to address class imbalance. 
                If a single value is provided, it is assumed to be the weight for the positive class, and the weight for the negative class is calculated as 1 - alpha. 
                If a list is provided, it should contain weight factors for each class.
            gamma (float, optional): The focusing parameter to control the degree of adjustment for misclassified samples. 
                Higher values of gamma give more weight to hard-to-classify examples, reducing the influence of easy examples. Defaults to 2.
            device (str, optional): The device on which to perform calculations ('cuda' or 'cpu'). Defaults to 'cuda'.

        Attributes:
            alpha (torch.Tensor): The calculated weight factors for each class.
            gamma (float): The focusing parameter.
            device (str): The device on which calculations are performed.

        Methods:
            forward(inputs, targets): Compute the Focal Loss given the input predictions and target labels.

        Note:
            - The Focal Loss is a modification of the standard cross-entropy loss that down-weights well-classified examples, focusing instead on hard-to-classify examples.
            - Ensure that the inputs are logits or unnormalized probabilities, and the targets are class labels.
            - This implementation supports binary or multiclass classification tasks.
        """

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

        if isinstance(alpha,(float,int)): 
            self.alpha = torch.Tensor([alpha,1-alpha]).to(device)

        if isinstance(alpha,(list, np.ndarray)): 
            self.alpha = torch.Tensor(alpha).to(device)


    def forward(self, inputs, targets):
        """
        Compute the Focal Loss given the input predictions and target labels.

        Args:
            inputs (torch.Tensor): The input predictions or logits from the model.
            targets (torch.Tensor): The target class labels.

        Returns:
            torch.Tensor: The computed Focal Loss value.

        Note:
            - Ensure that the inputs are logits or unnormalized probabilities, and the targets are class labels.
        """

        ce_loss = F.cross_entropy(inputs, targets, reduction='none').to(self.device)
        pt = torch.exp(-ce_loss).to(self.device)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
