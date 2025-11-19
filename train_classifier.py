import os
import numpy as np
from PIL import Image

# --- Deep Learning Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import timm

# --- Augmentation Imports ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda"
DATA_PATH = './vehicle_dataset'
MODEL_NAME = 'mobilenetv3_large_100'
NUM_CLASSES = 6 
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 25

# ==========================================
# 2. DATA TRANSFORMS
# ==========================================
# Rules to prepare images for the AI
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(),     # Randomly flip (Data Augmentation)
    A.ColorJitter(p=0.2),   # Randomly change brightness
    A.Normalize(),          # Standardize colors
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# ==========================================
# 3. CUSTOM DATASET CLASS
# ==========================================
class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        
        # Fix for nested folder structures (e.g., vehicle_dataset/data/...)
        data_root = os.path.join(root_dir, 'data')
        if not os.path.exists(data_root):
             data_root = root_dir 
            
        self.dataset = ImageFolder(root=data_root)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        
        # Load image and convert to RGB (Standard)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Apply transforms (Resize, Augment, Tensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_model():
    print(f"\n Starting training on: {DEVICE.upper()}")
    
    # --- A. Setup Data ---
    full_data = VehicleDataset(root_dir=DATA_PATH)
    
    # Save class names for the App
    print(f" Classes found: {full_data.classes}")
    with open("vehicle_classes.txt", "w") as f:
        f.write("\n".join(full_data.classes))

    # Split: 90% Training, 10% Validation
    train_len = int(0.9 * len(full_data))
    val_len = len(full_data) - train_len
    train_idx, val_idx = random_split(range(len(full_data)), [train_len, val_len])

    # Create Datasets
    train_ds = VehicleDataset(DATA_PATH, transform=train_transforms)
    val_ds = VehicleDataset(DATA_PATH, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(torch.utils.data.Subset(train_ds, train_idx), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), 
                            batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # --- B. Setup Model ---
    print(f" Loading Model: {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0

    # --- C. Training Loop ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 1. Train
        model.train()
        train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (i+1) % 10 == 0: 
                print(f"  Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # 2. Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 3. Statistics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f" Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2f}%")

        # 4. Save Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'vehicle_classifier.pth')
            print(f" Saved New Best Model! ({accuracy:.2f}%)")

    print("\n Training Complete.")

if __name__ == '__main__':
    train_model()