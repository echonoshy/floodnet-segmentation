import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.backends import cudnn

from core.dataset import FloodNetDataset
from models.unet import UNet 
from torchvision.models.segmentation import deeplabv3_resnet50
import segmentation_models_pytorch as smp

def set_model(opt):
    """
    Initialize the model, loss functions, and optimizer based on the provided options.
    Selects the model type based on configuration and loads pretrained weights if specified.
    """
    model = None
    if opt.name_net == 'unet':
        model = UNet(output_ch=opt.num_classes)
    elif opt.name_net == 'pspnet':
        model = smp.PSPNet('resnet34', in_channels=3, classes=opt.num_classes)
    elif opt.name_net == 'deeplab':
        model = deeplabv3_resnet50(num_classes=opt.num_classes)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate)

    # Load model from checkpoint if specified in options
    if opt.load_saved_model:
        path = os.path.join(opt.path_to_pretrained_model, opt.name_net + '_best.pth')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])

    # Enable CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, optimizer

def set_loader(opt):
    """
    Initialize data loaders for training and validation datasets with appropriate transformations.
    """
    # Define transformations for training and validation sets
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Resize((opt.resize_height, opt.resize_width)),
        T.ToTensor(),
        T.Normalize(mean=opt.mean, std=opt.std)
    ])

    val_transform = T.Compose([
        T.Resize((opt.resize_height, opt.resize_width)),
        T.ToTensor(),
        T.Normalize(mean=opt.mean, std=opt.std)
    ])

    # Transformations specific to target data in training and validation
    train_target_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Resize((opt.resize_height, opt.resize_width)),
        T.PILToTensor(),
    ])

    val_target_transform = T.Compose([
        T.Resize((opt.resize_height, opt.resize_width)),
        T.PILToTensor(),
    ])

    # Set paths and create dataset objects
    train_dir = os.path.join(opt.data_folder, 'train')
    val_dir = os.path.join(opt.data_folder, 'val')

    train_dataset = FloodNetDataset(train_dir, transform=train_transform, target_transform=train_target_transform)
    validation_dataset = FloodNetDataset(val_dir, transform=val_transform, target_transform=val_target_transform)

    # Data loaders for handling batch processing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_loader, val_loader

def save_model(model, optimizer, opt, epoch, save_file):
    """
    Save the model state, optimizer state, and other details to a file.
    """
    print('start to save model ... ')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
