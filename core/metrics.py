import numpy as np
from sklearn.metrics import confusion_matrix

def dice_coef(y_pred, y_true, epsilon=1e-7):
    """
    Calculate the Dice Coefficient for each class in the predictions and ground truth.

    Parameters:
        y_pred (tensor): The predicted segmentation map.
        y_true (tensor): The ground truth segmentation map.
        epsilon (float): A small number to avoid division by zero in calculations.

    Returns:
        float: The average Dice Coefficient across all classes except the background.
    """
    dice_scores = []

    # Determine the number of classes
    classes = int(np.max(y_true.detach().cpu().numpy())) + 1
    
    # Compute Dice Coefficient for each class
    for num_class in range(1, classes):
        target = (y_true == num_class)  # Ground truth for current class
        pred = (y_pred == num_class)    # Predictions for current class
        
        intersect = (target * pred).sum()  # Intersection
        union = target.sum() + pred.sum()  # Union

        # Calculate Dice score for the current class
        score = (2 * intersect + epsilon) / (union + epsilon)
        dice_scores.append(score)

    # Return the average Dice Coefficient across all classes
    return (sum(dice_scores) / len(dice_scores)).item()

def miou_coef(y_pred, y_true, num_classes=10):
    """
    Calculate the Mean Intersection over Union (mIoU) across all classes.

    Parameters:
        y_pred (tensor): The predicted segmentation map.
        y_true (tensor): The ground truth segmentation map.
        num_classes (int): The number of classes in the dataset.

    Returns:
        float: The mean IoU score across all classes.
    """
    # Flatten tensors and convert to numpy arrays
    y_pred_np = y_pred.flatten().detach().cpu().numpy()
    y_true_np = y_true.flatten().detach().cpu().numpy()
    
    # Compute the confusion matrix
    cf_matrix = confusion_matrix(y_true_np, y_pred_np, labels=np.arange(num_classes))
    
    # Calculate the intersection (diagonal elements) and union
    intersection = np.diag(cf_matrix)
    union = np.sum(cf_matrix, axis=1) + np.sum(cf_matrix, axis=0) - intersection

    # Filter out invalid classes (where the union is zero)
    valid_classes = union > 0
    iou_per_class = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=valid_classes)
    
    # Return the mean IoU across all valid classes
    return np.mean(iou_per_class[valid_classes])
