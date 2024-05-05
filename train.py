import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from core.metrics import dice_coef
from core.model import set_model, set_loader, save_model
from core.opt import opt
from core.logger import setup_logging

# Set up logging
logger = setup_logging(logger_name=f"train_{opt.name_net}")

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, epoch, opt):
    """
    Train the model for one epoch.

    Parameters:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for the training.
        epoch (int): Current training epoch.
        opt (Namespace): Training configuration options.
    
    Returns:
        float: Average training loss for the epoch.
        float: Average validation loss after the epoch.
        float: Validation Dice score after the epoch.
    """
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
                logger.info(
                    f"Train Epoch: [{epoch}/{opt.epochs}], lr: {optimizer.param_groups[0]['lr']:.6f}, Loss: {total_loss / total_num:.4f}"
                )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

    # Perform garbage collection and clear GPU cache
    torch.cuda.empty_cache()

    try:
        val_loss, val_dice = val(model, val_loader, criterion, opt)
        logger.info(f"Epoch {epoch}: Training Loss: {total_loss / total_num:.4f}, Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise

    return total_loss / total_num, val_loss, val_dice

# Validation function
def val(model, val_loader, criterion, opt):
    """
    Evaluate the model on the validation set.

    Parameters:
        model (torch.nn.Module): Model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        opt (Namespace): Configuration options.
    
    Returns:
        float: Average validation loss.
        float: Validation Dice score.
    """
    model.eval()
    val_loss, val_dice, total_num = 0, 0, 0

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
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Dice Coefficient: {val_dice:.4f}")
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise

    return val_loss, val_dice

# Plot generation function
def save_plots(train_loss_values, val_loss_values, val_dices, opt):
    """
    Generate plots for training and validation losses and Dice scores.

    Parameters:
        train_loss_values (list): List of training losses.
        val_loss_values (list): List of validation losses.
        val_dices (list): List of validation Dice scores.
        opt (Namespace): Configuration options.
    """
    try:
        plt.figure(figsize=(15, 10))
        plt.plot(train_loss_values, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.savefig(os.path.join(opt.results_folder, f"{opt.name_net}_train_loss.png"))
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.plot(val_loss_values, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Time")
        plt.legend()
        plt.savefig(os.path.join(opt.results_folder, f"{opt.name_net}_val_loss.png"))
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.plot(val_dices, label="Validation Dice")
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coefficient")
        plt.title("Validation Dice Over Time")
        plt.legend()
        plt.savefig(os.path.join(opt.results_folder, f"{opt.name_net}_val_dice.png"))
        plt.close()
    except Exception as e:
        logger.error(f"Error during plotting: {e}")
        raise

# Run training and evaluation
def run():
    """
    Run the training process, save the best model, and generate plots.
    """
    try:
        train_loader, val_loader, _ = set_loader(opt)
        model, criterion, optimizer = set_model(opt)
        save_file = os.path.join(opt.results_folder, opt.name_net + "_last.pth")

        logger.info("Start training...")
        best_val_dice = opt.threshold_val_dice
        train_loss_values, val_loss_values, val_dices = [], [], []

        for epoch in range(1, opt.epochs + 1):
            train_loss, val_loss, val_dice = train(model, train_loader, val_loader, criterion, optimizer, epoch, opt)
            train_loss_values.append(train_loss)
            val_loss_values.append(val_loss)
            val_dices.append(val_dice)

            if val_dice > best_val_dice:
                logger.info(f"Saving/updating current best model at epoch={epoch}")
                save_model(model, optimizer, opt, epoch, os.path.join(opt.results_folder, opt.name_net + "_best.pth"))
                best_val_dice = val_dice

            save_model(model, optimizer, opt, epoch, save_file)

        save_plots(train_loss_values, val_loss_values, val_dices, opt)
    except Exception as e:
        logger.error(f"Unhandled error in run function: {e}")
        raise

if __name__ == "__main__":
    run()
