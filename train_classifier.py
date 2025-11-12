import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
import os
import numpy as np
from PIL import Image

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = './vehicle_dataset'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 25
IMG_SIZE = 224
MODEL_NAME = 'mobilenetv3_large_100' # Fast and effective
NUM_CLASSES = 6 # *** THIS IS THE CORRECTED VALUE for the new dataset ***

# --- Albumentations Transforms ---
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- Custom Dataset for Albumentations ---
class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Use ImageFolder to find classes and files
        # The new dataset may have train/test splits, let's point to the main folder
        # ImageFolder will automatically find subdirs like 'Car', 'Bus', etc.
        data_root = os.path.join(root_dir, 'data') # Adjust if structure is different
        if not os.path.exists(data_root):
             data_root = root_dir # Fallback if 'dataset' subdir doesn't exist
            
        self.dataset = ImageFolder(root=data_root)
        self.samples = self.dataset.samples
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        
        # Read image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, target

# --- Main Training Function ---
def train_model():
    print(f"Using device: {DEVICE}")
    print(f"Loading data from: {DATA_PATH}")

    # We will manually split the main dataset.
    full_dataset = VehicleDataset(root_dir=DATA_PATH, transform=None)
    
    # Get class names and save them
    class_names = full_dataset.classes
    print(f"Found classes: {class_names}")
    with open("vehicle_classes.txt", "w") as f:
        f.write("\n".join(class_names))

    # Create a 90/10 train/val split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])

    # Create new dataset instances with the *same* underlying data but different indices and transforms
    train_dataset = VehicleDataset(root_dir=DATA_PATH, transform=train_transforms)
    val_dataset = VehicleDataset(root_dir=DATA_PATH, transform=val_transforms)

    # Use Subset to apply the indices
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Load model from timm
    print(f"Loading model: {MODEL_NAME}")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 50 == 0: # Print every 50 batches
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'vehicle_classifier.pth')
            print(f"  ** New best model saved with {val_acc:.2f}% accuracy **")

    print("\nClassifier training complete.")
    print(f"Best model saved to 'vehicle_classifier.pth' with {best_val_acc:.2f}% accuracy.")

if __name__ == '__main__':
    train_model()