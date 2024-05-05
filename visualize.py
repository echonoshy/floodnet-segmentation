import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision import transforms as T
from core.model import set_model
from core.opt import opt

# Define a custom colormap for semantic classes
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

# Function to unnormalize a tensor image
def unnormalize(tensor, mean=[-0.2417, 0.8531, 0.1789], std=[0.9023, 1.1647, 1.3271]):
    mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    return tensor * std + mean

# Load the model based on the specified name
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

# Visualize predictions made by the model
def visualize_predictions(model, image_paths, mask_paths, opt=opt):
    """
    Generate and display visual predictions for a list of test images.

    Parameters:
        model (torch.nn.Module): Trained model to use for predictions.
        image_paths (list): List of file paths for test images.
        mask_paths (list): List of file paths for the ground truth masks.
        opt (Namespace): Configuration options.
    """
    # Define image and mask transformations
    img_transform = T.Compose([
        T.Resize((opt.resize_height, opt.resize_width)),
        T.ToTensor(),
        T.Normalize(mean=opt.mean, std=opt.std)
    ])
    mask_transform = T.Compose([
        T.Resize((opt.resize_height, opt.resize_width)),
        T.PILToTensor()
    ])

    # Prepare subplots
    n = len(image_paths)
    fig, axs = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axs = [axs]

    with torch.no_grad():
        for i in range(n):
            # Load and preprocess the images and masks
            img = img_transform(Image.open(image_paths[i]))
            mask = mask_transform(Image.open(mask_paths[i]))

            # Move image to GPU if available
            if torch.cuda.is_available():
                img = img.cuda(non_blocking=True)

            # Make predictions using the model
            if opt.name_net == 'deeplab': 
                pred = model(img[None, :, :, :])['out']
            else:
                pred = model(img[None, :, :, :])
                
            pred = torch.squeeze(pred).argmax(0).squeeze()
            pred = pred.cpu().detach().numpy()

            # Unnormalize the image and display results
            axs[i][0].imshow(np.squeeze(unnormalize(np.transpose(img.squeeze().cpu(), (1, 2, 0)))))
            axs[i][1].imshow(mask.squeeze(), cmap=custom_cmap, vmin=0, vmax=9)
            axs[i][2].imshow(pred.squeeze(), cmap=custom_cmap, vmin=0, vmax=9)

            # Set subplot titles
            axs[i][0].set_title("Image")
            axs[i][1].set_title("Label")
            axs[i][2].set_title("Prediction")

    plt.tight_layout()
    os.makedirs("inference_images", exist_ok=True)
    plt.savefig(f'inference_images/test_{opt.name_net}.png')
    plt.show()

# Main script execution
if __name__ == '__main__':
    # Load the specified model
    activate_model = load_model(opt.name_net)

    # List of test images and corresponding ground truth masks
    image_paths = [
        '/root/autodl-tmp/test/6467.jpg',
        '/root/autodl-tmp/test/6691.jpg',
        '/root/autodl-tmp/test/6718.jpg',
        '/root/autodl-tmp/test/6927.jpg'
    ]
    mask_paths = [
        '/root/autodl-tmp/test/6467_lab.png',
        '/root/autodl-tmp/test/6691_lab.png',
        '/root/autodl-tmp/test/6718_lab.png',
        '/root/autodl-tmp/test/6927_lab.png'
    ]

    # Visualize predictions
    visualize_predictions(activate_model, image_paths, mask_paths)
