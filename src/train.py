import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms

from model import create_model
from data_preprocessing import create_data_loaders

class HandGestureDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None):
        self.data_dir = data_dir
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create gesture to index mapping
        self.gesture_to_idx = {
            'open': 0,
            'closed': 1,
            'pointing': 2,
            'pinching': 3,
            'waving': 4
        }
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self.key_points = []
        
        for img_name, data in self.annotations.items():
            self.image_paths.append(os.path.join(data_dir, img_name))
            self.labels.append(self.gesture_to_idx[data['gesture']])
            self.key_points.append(data['key_points'])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert key points to tensor and pad to fixed size
        max_points = 20  # Maximum number of key points
        key_points = self.key_points[idx]
        padded_points = np.zeros((max_points, 3))
        
        # Fill with actual key points
        n_points = min(len(key_points), max_points)
        for i in range(n_points):
            padded_points[i] = [
                key_points[i]['x'],
                key_points[i]['y'],
                key_points[i]['depth']
            ]
        
        # Convert to tensor
        key_points_tensor = torch.tensor(padded_points, dtype=torch.float32)
        
        # Apply image transformations
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'key_points': key_points_tensor,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'valid_points': torch.tensor(n_points, dtype=torch.long)
        }

class HandGestureModel(nn.Module):
    def __init__(self, num_classes=5):
        super(HandGestureModel, self).__init__()
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # MLP for key points
        self.key_points_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, key_points, valid_points):
        # Process image
        batch_size = image.size(0)
        image_features = self.cnn(image).view(batch_size, -1)
        
        # Process key points
        key_points_features = self.key_points_mlp(key_points)
        
        # Mask out padding
        mask = torch.arange(key_points.size(1), device=key_points.device)[None, :] < valid_points[:, None]
        key_points_features = key_points_features * mask.unsqueeze(-1)
        
        # Average pooling over valid points
        key_points_features = key_points_features.sum(dim=1) / valid_points.float().unsqueeze(-1)
        
        # Combine features
        combined = torch.cat([image_features, key_points_features], dim=1)
        
        # Classify
        return self.classifier(combined)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = batch['image'].to(device)
            key_points = batch['key_points'].to(device)
            valid_points = batch['valid_points'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, key_points, valid_points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = batch['image'].to(device)
                key_points = batch['key_points'].to(device)
                valid_points = batch['valid_points'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, key_points, valid_points)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/best_model.pth')
    
    return train_losses, val_losses, train_accs, val_accs

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = HandGestureDataset(
        annotations_file='data/processed/hand_features.json',
        data_dir='data/raw'
    )
    
    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Create model
    model = HandGestureModel().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=50, device=device
    )
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

if __name__ == '__main__':
    main() 