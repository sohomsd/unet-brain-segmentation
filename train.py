import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Normalize
from tqdm import tqdm
import pickle

from data import BraTSDataset, get_distribution_stats
from unet_models.base import Unet
from unet_models.dilated_inception import DIUnet
from metrics import dice_coefficient, jaccard_index


def train_model(model:nn.Module, optimizer, train_loader:DataLoader, val_loader:DataLoader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    history = {"num_epochs": num_epochs, 
               "train_loss": [],
               "train_dice": [],
               "train_jaccard": [],
               "val_loss": [],
               "val_dice": [],
               "val_jaccard": []
               } 
    for epoch in range(num_epochs):
        # Train model using training set
        model.train()
        loss_sum = 0
        dice_sum = 0
        jaccard_sum = 0
        for image, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, torch.argmax(target, 1))
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item()
            dice_sum += dice_coefficient(output, target, device).detach().mean().item()
            jaccard_sum += jaccard_index(output, target, device).detach().mean().item()

        # Append training metrics
        history["train_loss"].append(loss_sum/len(train_loader))
        history["train_dice"].append(dice_sum/len(train_loader))
        history["train_jaccard"].append(jaccard_sum/len(train_loader))
        print(f"Training Loss: {(loss_sum/len(train_loader)):.4f}")
        print(f"Training Dice: {(dice_sum/len(train_loader)):.4f}")
        print(f"Training Jaccard: {(jaccard_sum/len(train_loader)):.4f}")

        model.eval()
        loss_sum = 0
        dice_sum = 0
        jaccard_sum = 0
        for image, target in val_loader:
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, torch.argmax(target, 1))

            loss_sum += loss.detach().item()
            dice_sum += dice_coefficient(output, target, device).detach().mean().item()
            jaccard_sum += jaccard_index(output, target, device).detach().mean().item()
        
        # Append validation metrics
        history["val_loss"].append(loss_sum/len(val_loader))
        history["val_dice"].append(dice_sum/len(val_loader))
        history["val_jaccard"].append(jaccard_sum/len(val_loader))
        print(f"Validation Loss: {(loss_sum/len(val_loader)):.4f}")
        print(f"Validation Dice: {(dice_sum/len(val_loader)):.4f}")
        print(f"Validation Jaccard: {(jaccard_sum/len(val_loader)):.4f}")

    return model.state_dict(), history

if __name__ == "__main__":
    torch.manual_seed(42)

    models = {"Unet": Unet(4, 4), 
              "DIUnet": DIUnet(4, 4)}
    
    
    data_root_dir = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    total_data = BraTSDataset(data_root_dir, slice=77)
    mean, std = get_distribution_stats(total_data)
    img_transform = Normalize(mean, std)

    train_img_dirs, test_img_dirs = train_test_split(os.listdir(data_root_dir), test_size=0.2, 
                                                     shuffle=True, random_state=42)
    train_data = BraTSDataset(data_root_dir, slice=77, img_dirs=train_img_dirs, img_transform=img_transform)
    val_data = BraTSDataset(data_root_dir, slice=77, img_dirs=test_img_dirs, img_transform=img_transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

    for model_name, model in models.items():
        print(f"Training {model_name}:\n")

        optimizer = Adam(model.parameters())
        model_weights, history = train_model(model, optimizer, train_loader, val_loader, num_epochs=25)

        weights_dir = "results/model_weights"
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        torch.save(model_weights, os.path.join(weights_dir, f"{model_name}_weights.pt"))

        history_dir = "results/training_history"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        with open(os.path.join(history_dir, f"{model_name}_history.pkl"), "wb") as f:
            pickle.dump(history, f)