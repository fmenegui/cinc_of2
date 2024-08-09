import torch
import torch.nn.functional as F
import torch.nn as nn

class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward(self, logits, labels):
        """
        Compute the binary cross-entropy loss.

        Args:
            logits (torch.Tensor): Logits from the model (pre-sigmoid).
            labels (torch.Tensor): Ground truth binary labels.

        Returns:
            torch.Tensor: Computed binary cross-entropy loss.
        """
        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Calculate the BCE Loss
        bce_loss = -labels * torch.log(probabilities + 1e-6) - (1 - labels) * torch.log(1 - probabilities + 1e-6)
        
        # Return the mean of the BCE loss
        return torch.mean(bce_loss)

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid activation to get [0,1] range
        predictions = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()                            
        dice = (2.*intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice
    
    import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftF1Loss(nn.Module):
    def __init__(self):
        super(SoftF1Loss, self).__init__()

    def forward(self, logits, labels):
        """
        Calculate soft F1 Score as a loss.

        Args:
            logits (torch.Tensor): Model predictions as raw logits for binary classification.
            labels (torch.Tensor): Ground truth labels, same shape as logits.

        Returns:
            torch.Tensor: Computed soft F1 loss.
        """
        # Apply sigmoid to logits to get the probabilities
        probs = torch.sigmoid(logits)

        # Calculate true positives, false positives, and false negatives
        tp = torch.sum(labels * probs, dim=0)
        fp = torch.sum((1 - labels) * probs, dim=0)
        fn = torch.sum(labels * (1 - probs), dim=0)

        # Calculate Precision and Recall
        precision = tp / (tp + fp + 1e-6)  # add small epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-6)

        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Calculate F1 Loss
        f1_loss = 1 - f1
        return f1_loss.mean()  # return the average over all classes in the batch


class MCC_Loss(nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        inputs = torch.sigmoid(inputs)
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha (float): Balancing factor, default 0.25.
            gamma (float): Focusing parameter, default 2.0.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.

        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid).
            targets (torch.Tensor): Ground truth labels, with the same shape as inputs.
        
        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute the loss components
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEF1Loss(nn.Module):
    def __init__(self, beta=1, weight_bce=0.5, weight_f1=0.5):
        super().__init__()
        self.beta = beta
        self.weight_bce = weight_bce
        self.weight_f1 = weight_f1
    
    def forward(self, logits, targets):
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Soft predictions
        probs = torch.sigmoid(logits)
        
        # True Positives, False Positives, False Negatives
        tp = (probs * targets).sum(dim=0)
        fp = (probs * (1 - targets)).sum(dim=0)
        fn = ((1 - probs) * targets).sum(dim=0)
        
        # Precision and Recall calculation
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # F1 Score
        f1_score = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + 1e-8)
        f1_loss = 1 - f1_score.mean()
        
        # Combined loss
        return self.weight_bce * bce_loss + self.weight_f1 * f1_loss


class BCEMCCLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_mcc=0.5):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_mcc = weight_mcc

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        inputs = torch.sigmoid(inputs)
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        mcc_loss = 1 - mcc
        return self.weight_bce * bce_loss + self.weight_mcc * mcc_loss
