import os
import torch
from core.model import set_model, set_loader
from core.metrics import dice_coef, miou_coef
from core.opt import opt


def evaluate(model, test_loader):
    """
    Evaluate the model and print the average Dice and mIoU scores.

    Parameters:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    """
    model.eval()  
    dice_scores, miou_scores = [], []  

    # Disable gradient calculations to save memory and improve performance
    with torch.no_grad():
        for image, mask in test_loader:
            
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

            # Make predictions using the model
            output = model(image) if opt.name_net != 'deeplab' else model(image)['out']
            pred = output.argmax(dim=1)  # Get the predicted class for each pixel
            mask = torch.squeeze(mask, dim=1)  # Remove extra dimensions

            # Compute Dice and mIoU for the current batch
            dice = dice_coef(pred, mask)
            miou = miou_coef(pred, mask)

            dice_scores.append(dice)
            miou_scores.append(miou)

    # Calculate the average scores for all batches
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_miou = sum(miou_scores) / len(miou_scores)
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average mIoU: {avg_miou:.4f}")


def main():
    """
    Load the model and perform evaluation.
    """
    # Load the pre-trained model
    model, _, _ = set_model(opt)
    model_path = os.path.join(opt.results_folder, f"{opt.name_net}_best.pth")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])  
    model.eval()  
    
    if torch.cuda.is_available():
        model = model.cuda()

    # Get the test DataLoader
    _, _, test_loader = set_loader(opt)

    # Evaluate the model performance
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
