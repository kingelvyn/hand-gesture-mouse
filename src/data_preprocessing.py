import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

class HandGestureDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None, max_key_points=21):
        """
        Args:
            data_dir (str): Directory with all the images
            annotations_file (str): Path to the JSON file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
            max_key_points (int): Maximum number of key points to pad to
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_key_points = max_key_points
        
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
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get image filename and annotations
        img_name = list(self.annotations.keys())[idx]
        img_data = self.annotations[img_name]
        
        # Load image
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get gesture label
        gesture = img_data['gesture']
        label = self.gesture_to_idx[gesture]
        
        # Get key points
        key_points = img_data['key_points']
        
        # Convert key points to tensor with padding
        key_points_tensor = torch.zeros((self.max_key_points, 3))  # x, y, depth
        for i, point in enumerate(key_points[:self.max_key_points]):
            key_points_tensor[i] = torch.tensor([
                point['x'],
                point['y'],
                point['depth']
            ])
        
        # Create a mask for valid key points
        valid_points = torch.zeros(self.max_key_points, dtype=torch.bool)
        valid_points[:len(key_points)] = True
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'key_points': key_points_tensor,
            'valid_points': valid_points,
            'label': label,
            'gesture': gesture
        }

def get_transforms(is_training=True):
    """Get transforms for training or validation."""
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(data_dir, annotations_file, batch_size=32, train_split=0.8):
    """Create training and validation data loaders."""
    # Create full dataset
    full_dataset = HandGestureDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        transform=get_transforms(is_training=True)
    )
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the data loading
    data_dir = "data/raw"
    annotations_file = "data/processed/hand_features.json"
    
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        annotations_file=annotations_file,
        batch_size=8
    )
    
    # Print dataset information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    print("\nSample batch keys:", sample_batch.keys())
    print("Image shape:", sample_batch['image'].shape)
    print("Key points shape:", sample_batch['key_points'].shape)
    print("Valid points shape:", sample_batch['valid_points'].shape)
    print("Labels:", sample_batch['label'])
    print("Gestures:", sample_batch['gesture']) 