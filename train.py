from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, Resize

from data import BraTSDataset
from unet_models.base import Unet
from unet_models.dilated_inception import DIUnet


def train_model(model:nn.Module, optimizer, train_loader:DataLoader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    train_loss = []
    for epoch in range(num_epochs):
        # Train model using training set
        model.train()
        loss_sum = 0
        for image, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, torch.argmax(target, 1))
            loss.backward()
            optimizer.step()

            loss_sum += loss
        # Append average training loss to train_loss list
        train_loss.append(loss_sum/len(train_loader))
        print(f"Training Loss: {(loss_sum/len(train_loader)):.4f}")

    return model.state_dict(), train_loss

if __name__ == "__main__":
    torch.manual_seed(42)

    models = {"Unet": Unet(4, 4), 
              "DIUnet": DIUnet(4, 4)}
    

    img_transform = Compose([
        Resize((128, 128))
    ])
    
    data_root_dir = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    train_data = BraTSDataset(data_root_dir, slice=77, 
                              contrast_transform=img_transform, seg_transform=img_transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    for model_name, model in models.items():
        print(f"Training {model_name}:\n")

        optimizer = Adam(model.parameters())
        model_weights, train_loss = train_model(model, optimizer, train_loader, num_epochs=20)

        torch.save(model_weights, f"results/model_weights/{model_name}_weights.pt")

        with open(f"results/training_loss/{model_name}_trainingloss.pkl", "wb") as f:
            pickle.dump(train_loss, f)