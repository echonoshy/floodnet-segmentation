import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class FloodNetDataset(Dataset):
    """
    Dataset class for FloodNet, a dataset for image segmentation tasks related to flood scenarios.
    classes:
            
        "background",
        "building-flooded",
        "building non-flooded",
        "road flooded",
        "road non-flooded",
        "water",
        "tree",
        "vehicle",
        "pool",
        "grass"
            
    Attributes:
        base_folder (str): The directory containing the training images.
        transform (callable): Function to apply transformations on the images.
        target_transform (callable): Function to apply transformations on the labels (segmentation masks).
    """

    def __init__(self, base_folder="./data/train", transform=lambda x: x, target_transform=lambda y: y):
        """
        Parameters:
            base_folder (str): The path to the folder containing image files.
            transform (callable): The function to transform images. 
            target_transform (callable): The function to transform labels.
        """
        super().__init__()
        self.base_folder = base_folder
        self.im_files = [f for f in os.listdir(self.base_folder) if f.endswith(".jpg")]  # List of image files
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.im_files)

    def __getitem__(self, index):
        """
        Fetches the image and its corresponding label at the specified index.

        Parameters:
            index (int): The index of the item.

        Returns:
            tuple: Contains the transformed image and label.
        """
        img_file = os.path.join(self.base_folder, self.im_files[index])  # Path to the image file
        gt_file = img_file.replace(".jpg", "_lab.png")  # Corresponding ground truth label file path
        img = Image.open(img_file)  # Open the image file
        label = Image.open(gt_file)  # Open the ground truth label file

        # Apply transformations
        state = torch.get_rng_state()  # Save the random state
        img = self.transform(img)
        torch.set_rng_state(state)  # Ensure same random state is used for both images and labels
        label = self.target_transform(label)

        return img, label
