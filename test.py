import os
import torch
import numpy as np
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix
from core.model import set_model
from core.opt import opt

# Load the model
def load_model():
    model, _, _ = set_model(opt)
    model_path = os.path.join(opt.results_folder, f"{opt.name_net}_best.pth")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# Define transformations for test images and labels
img_transform = T.Compose([
    T.Resize((opt.resize_height, opt.resize_width)),
    T.ToTensor(),
    T.Normalize(mean=opt.mean, std=opt.std)
])
mask_transform = T.Compose([
    T.Resize((opt.resize_height, opt.resize_width)),
    T.PILToTensor()
])

# Compute Dice coefficient for each class
def compute_dice(y_true, y_pred):
    dice_scores = []
    for class_id in np.unique(y_true):
        true_class = (y_true == class_id)
        pred_class = (y_pred == class_id)
        intersection = np.sum(true_class * pred_class)
        dice_score = (2 * intersection + 1e-6) / (np.sum(true_class) + np.sum(pred_class) + 1e-6)
        dice_scores.append(dice_score)
    return np.mean(dice_scores)

# Compute mIoU for all classes
def compute_miou(y_true, y_pred, num_classes):
    cf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    intersection = np.diag(cf_matrix)
    union = np.sum(cf_matrix, axis=1) + np.sum(cf_matrix, axis=0) - intersection
    miou_per_class = intersection / (union + 1e-6)
    return np.mean(miou_per_class)

# Load test images and calculate metrics
def evaluate(model, test_folder, num_classes):
    dice_scores, miou_scores = [], []
    image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    for img_file in image_files:
        img_path = os.path.join(test_folder, img_file)
        mask_path = os.path.join(test_folder, img_file.replace('.jpg', '_lab.png'))
        
        # Load and transform image and mask
        img = img_transform(Image.open(img_path)).unsqueeze(0)
        mask = mask_transform(Image.open(mask_path)).squeeze()
        
        if torch.cuda.is_available():
            img = img.cuda()
        
        # Get prediction from the model
        with torch.no_grad():
            pred = model(img)['out'] if hasattr(model, 'aux_classifier') else model(img)
            pred = torch.squeeze(pred).argmax(dim=0).cpu().detach().numpy()
        
        mask = np.array(mask)
        
        # Compute metrics
        dice = compute_dice(mask, pred)
        dice_scores.append(dice)
        miou = compute_miou(mask.flatten(), pred.flatten(), num_classes)
        miou_scores.append(miou)
    
    # Calculate average scores
    avg_dice = np.mean(dice_scores)
    avg_miou = np.mean(miou_scores)
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average mIoU: {avg_miou:.4f}")

# Main function to run the evaluation
def main():
    model = load_model()
    test_folder = os.path.join(opt.data_folder, 'test')  # Modify if necessary
    num_classes = opt.num_classes
    evaluate(model, test_folder, num_classes)

if __name__ == "__main__":
    main()
