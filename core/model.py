import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.backends import cudnn

# Model-specific imports could be something like:
# from models.unet import U_Net
# import segmentation_models_pytorch as smp

def set_model(opt):
    """
    Initialize the model, loss functions, and optimizer based on the provided options.
    """
    model = None
    if opt.name_net == 'unet':
        model = U_Net(output_ch=opt.num_classes)
    elif opt.name_net == 'pspnet':
        model = smp.PSPNet('resnet34', in_channels=3, classes=opt.num_classes)
    elif opt.name_net == 'deeplab':
        model = deeplabv3_resnet50(num_classes=opt.num_classes)

    criterion = nn.CrossEntropyLoss()
    criterion_psl = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate)

    if opt.load_saved_model:
        path = os.path.join(opt.path_to_pretrained_model, opt.name_net + '_best.pth')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_psl = criterion_psl.cuda()
        cudnn.benchmark = True

    return model, criterion, criterion_psl, optimizer

def set_loader(opt):
    """
    Initialize data loaders for training and validation datasets.
    """
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

    train_dir = os.path.join(opt.labeled_data_folder, 'train')
    val_dir = os.path.join(opt.labeled_data_folder, 'val')

    train_dataset = FloodNetDataset_Labeled(train_dir, transform=train_transform, target_transform=train_target_transform)
    validation_dataset = FloodNetDataset_Labeled(val_dir, transform=val_transform, target_transform=val_target_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_loader, val_loader

def save_model(model, optimizer, opt, epoch, save_file):
    """
    Save the model state, optimizer state, and other details to a file.
    """
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
