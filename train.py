import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import gc
from torch.backends import cudnn

from core.dataset import FloodNetDataset
from core.metrics import dice_coef
from core.model import set_model, set_loader, save_model
from core.opt import opt


def train(model, train_loader, val_loader, criterion, optimizer, epoch, opt):
    model.train()
    total_loss, total_num = 0.0, 0
    
    try:
        for idx, (image, mask) in enumerate(train_loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

            output = model(image) if opt.name_net != 'deeplab' else model(image)['out']

            mask = torch.squeeze(mask, dim=1)
            loss = criterion(output, mask.long())
            loss.backward()
            optimizer.step()
            total_num += mask.size(0)
            total_loss += loss.item() * mask.size(0)

            if (idx + 1) % opt.print_freq == 0:
                print(f'Fully_supervised-Train Epoch: [{epoch}/{opt.epochs}], lr: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {total_loss / total_num:.4f}')
                sys.stdout.flush()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    gc.collect()
    torch.cuda.empty_cache()
    val_loss, val_dice = val(model, val_loader, criterion, opt)
    print("Epoch total loss", total_loss / total_num)
    train_loss = total_loss / total_num
    return train_loss, val_loss, val_dice 

def val(model, val_loader, criterion, opt):
    model.eval()
    val_loss, val_dice = 0, 0
    total_num = 0
    try:
        with torch.no_grad():
            for image, mask in val_loader:
                if torch.cuda.is_available():
                    image = image.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)

                output = model(image) if opt.name_net != 'deeplab' else model(image)['out']
                
                mask = torch.squeeze(mask, dim=1)
                loss = criterion(output, mask.long())
                total_num += mask.size(0)
                val_loss += loss.item() * mask.size(0)
                dice = dice_coef(output.argmax(dim=1), mask)
                val_dice += dice

        val_loss /= total_num
        val_dice /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}, Validation DICE coefficient: {val_dice:.4f}")
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)
    return val_loss, val_dice

def save_plots(train_loss_values, val_loss_values, val_dices, opt):
    try:
        plt.figure(figsize=(15, 10))
        plt.plot(train_loss_values, label='Train loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig(os.path.join(opt.results_folder, f'{opt.name_net}_train_loss.png'))
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.plot(val_loss_values, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss Over Time')
        plt.legend()
        plt.savefig(os.path.join(opt.results_folder, f'{opt.name_net}_val_loss.png'))
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.plot(val_dices, label='Validation Dice')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Coefficient')
        plt.title('Validation Dice Over Time')
        plt.legend()
        plt.savefig(os.path.join(opt.results_folder, f'{opt.name_net}_val_dice.png'))
        plt.close()
    except Exception as e:
        print(f"Error during plotting: {e}")
        sys.exit(1)

def run():
    try:
        train_loader, val_loader = set_loader(opt)
        model, criterion, optimizer = set_model(opt)
        save_file = os.path.join(opt.results_folder, opt.name_net + '_last.pth')

        print("Start training ...")
        best_val_dice = opt.threshold_val_dice
        train_loss_values, val_loss_values, val_dices = [], [], []

        for epoch in range(1, opt.epochs + 1):
            train_loss, val_loss, val_dice = train(model, train_loader, val_loader, criterion, optimizer, epoch, opt)
            train_loss_values.append(train_loss)
            val_loss_values.append(val_loss)
            val_dices.append(val_dice)

            if val_dice > best_val_dice:
                print(f"Saving/updating current best model at epoch={epoch}")
                save_model(model, optimizer, opt, epoch, os.path.join(opt.results_folder, opt.name_net + '_best.pth'))
                best_val_dice = val_dice

            save_model(model, optimizer, opt, epoch, save_file)

        save_plots(train_loss_values, val_loss_values, val_dices, opt)
    except Exception as e:
        print(f"Unhandled error in run function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
