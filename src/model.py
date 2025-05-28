import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class HandGestureModel(nn.Module):
    def __init__(self, num_classes=5):
        """
        Initialize the Hand Gesture Recognition model.
        
        Args:
            num_classes (int): Number of gesture classes to predict
        """
        super(HandGestureModel, self).__init__()
        
        # Image processing branch (CNN with residual blocks)
        self.image_branch = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Key points processing branch (MLP with residual connections)
        self.keypoints_branch = nn.Sequential(
            nn.Linear(21 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, key_points, valid_points=None):
        """
        Forward pass of the model.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224)
            key_points (torch.Tensor): Key points tensor of shape (batch_size, 21, 3)
            valid_points (torch.Tensor, optional): Boolean mask for valid key points
            
        Returns:
            torch.Tensor: Class predictions
        """
        # Process image
        image_features = self.image_branch(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Process key points
        key_points = key_points.view(key_points.size(0), -1)  # Flatten key points
        keypoint_features = self.keypoints_branch(key_points)
        
        # Combine features
        combined_features = torch.cat([image_features, keypoint_features], dim=1)
        
        # Get predictions
        predictions = self.classifier(combined_features)
        
        return predictions

def create_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create and initialize the model.
    
    Args:
        device (str): Device to place the model on
        
    Returns:
        HandGestureModel: Initialized model
    """
    model = HandGestureModel()
    model = model.to(device)
    return model

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)
    
    # Create dummy input
    batch_size = 8
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_keypoints = torch.randn(batch_size, 21, 3).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_image, dummy_keypoints)
    
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}") 