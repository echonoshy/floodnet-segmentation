import numpy as np

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
    dice_scores = []  # List to store the Dice Coefficients for each class

    # Determine the number of classes 
    classes = int(np.max(y_true.detach().cpu().numpy())) + 1
    
    # Compute Dice Coefficient for each class 
    for num_class in range(1, classes):
        target = (y_true == num_class)  # Ground truth for current class
        pred = (y_pred == num_class)  # Predictions for current class
        
        intersect = (target & pred).sum()  # Intersection of prediction and ground truth
        union = target.sum() + pred.sum()  # Union of prediction and ground truth

        # Dice score calculation for current class
        score = (2 * intersect + epsilon) / (union + epsilon)
        dice_scores.append(score)  # Append the score for this class

    # Return the average Dice Coefficient across all classes
    # return np.mean(dice_scores).item() if dice_scores else 0
    return (sum(dice_scores) / len(dice_scores)).item()
