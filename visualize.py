import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision import transforms as T
from core.model import set_model
from core.opt import opt

# Custom colormap and names for semantic classes
custom_colors = [
    [0, 0, 0],  # Background
    [1, 0, 0],  # Building flooded
    [1, 0.75, 0],  # Building non-flooded
    [0, 0.3, 0.7],  # Road flooded
    [0.6, 0.6, 0.6],  # Road non-flooded
    [0, 0.5, 1],  # Water
    [0, 0.85, 0],  # Tree
    [1, 0, 1],  # Vehicle
    [0.7, 0.7, 0.9],  # Pool
    [0.1, 0.55, 0.3]  # Grass
]
custom_cmap = ListedColormap(custom_colors)

# Unnormalization function
def unnormalize(tensor, mean=[-0.2417, 0.8531, 0.1789], std=[0.9023, 1.1647, 1.3271]):
    mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    return tensor * std + mean

# Load a specific model and weights based on name
def load_model(model_name):
    opt.name_net = model_name
    model, _, _ = set_model(opt)
    model_path = os.path.join(opt.results_folder, f"{model_name}_best.pth")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# Visualize predictions
def visualize_predictions(model, image_paths, mask_paths, img_size=(512, 512)):
    # Define image and mask transformations
    img_transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[-0.2417, 0.8531, 0.1789], std=[0.9023, 1.1647, 1.3271])
    ])
    mask_transform = T.Compose([
        T.Resize(img_size),
        T.PILToTensor()
    ])

    # Prepare subplots
    n = len(image_paths)
    fig, axs = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axs = [axs]

    with torch.no_grad():
        for i in range(n):
            # Load and process images/masks
            img = img_transform(Image.open(image_paths[i])).unsqueeze(0)
            mask = mask_transform(Image.open(mask_paths[i]))

            # Move image to GPU if available
            if torch.cuda.is_available():
                img = img.cuda()

            # Make predictions
            pred = model(img)['out'] if hasattr(model, 'aux_classifier') else model(img)
            pred = torch.squeeze(pred).argmax(dim=0).cpu().detach().numpy()

            # Unnormalize and visualize
            img_unnorm = unnormalize(img.squeeze().cpu(), mean=[-0.2417, 0.8531, 0.1789], std=[0.9023, 1.1647, 1.3271])
            axs[i][0].imshow(np.transpose(img_unnorm, (1, 2, 0)))
            axs[i][1].imshow(mask.squeeze(), cmap=custom_cmap, vmin=0, vmax=9)
            axs[i][2].imshow(pred, cmap=custom_cmap, vmin=0, vmax=9)

            # Set titles
            axs[i][0].set_title("Image")
            axs[i][1].set_title("Label")
            axs[i][2].set_title("Prediction")

    plt.tight_layout()
    plt.show()
    
# Visualize predictions of multiple models
def visualize_predictions_multiple(models, model_names, image_paths, mask_paths, img_size=(512, 512)):
    # Define image and mask transformations
    img_transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[-0.2417, 0.8531, 0.1789], std=[0.9023, 1.1647, 1.3271])
    ])
    mask_transform = T.Compose([
        T.Resize(img_size),
        T.PILToTensor()
    ])

    # Prepare subplots with extra columns for additional models
    n = len(image_paths)
    num_models = len(models)
    fig, axs = plt.subplots(n, 2 + num_models, figsize=(6 * (2 + num_models), 6 * n))
    if n == 1:
        axs = [axs]

    with torch.no_grad():
        for i in range(n):
            # Load and process images/masks
            img = img_transform(Image.open(image_paths[i])).unsqueeze(0)
            mask = mask_transform(Image.open(mask_paths[i]))

            # Move image to GPU if available
            if torch.cuda.is_available():
                img = img.cuda()

            # Unnormalize the input image for visualization
            img_unnorm = unnormalize(img.squeeze().cpu(), mean=[-0.2417, 0.8531, 0.1789], std=[0.9023, 1.1647, 1.3271])
            axs[i][0].imshow(np.transpose(img_unnorm, (1, 2, 0)))
            axs[i][1].imshow(mask.squeeze(), cmap=custom_cmap, vmin=0, vmax=9)
            
            # Predict and visualize for each model
            for j, model in enumerate(models):
                pred = model(img)['out'] if hasattr(model, 'aux_classifier') else model(img)
                pred = torch.squeeze(pred).argmax(dim=0).cpu().detach().numpy()
                axs[i][2 + j].imshow(pred, cmap=custom_cmap, vmin=0, vmax=9)
                axs[i][2 + j].set_title(model_names[j])

            # Set titles for input and label
            axs[i][0].set_title("Image")
            axs[i][1].set_title("Label")

    plt.tight_layout()
    plt.show()

# Example usage
model_unet = load_model('unet')
model_pspnet = load_model('pspnet')
model_deeplab = load_model('deeplab')

models = [model_unet, model_pspnet, model_deeplab]
model_names = ['U-Net', 'PSPNet', 'DeepLab']

image_paths = ['./data/val/6651.jpg', './data/val/7488.jpg', './data/val/6734.jpg']
mask_paths = ['./data/val/6651_lab.png', './data/val/7488_lab.png', './data/val/6734_lab.png']

visualize_predictions_multiple(models, model_names, image_paths, mask_paths)
