import cv2
import os
import json
import numpy as np
from pathlib import Path
import random

def load_annotations(annotations_file):
    """Load annotations from JSON file."""
    with open(annotations_file, 'r') as f:
        return json.load(f)

def visualize_annotations(image_path, key_points, save_path=None):
    """Visualize hand annotations on the image."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create skin mask
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
    # Draw key points
    for point in key_points:
        x = int(point['x'] * image.shape[1])
        y = int(point['y'] * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    
    # Add text showing number of key points
    cv2.putText(image, f"Key points: {len(key_points)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, image)
    
    return image

def main():
    # Load annotations
    annotations_file = 'data/processed/hand_features.json'
    annotations = load_annotations(annotations_file)
    
    # Create output directory
    output_dir = 'data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples from each gesture
    gestures = ['open', 'closed', 'pointing', 'pinching', 'waving']
    samples_per_gesture = 5
    
    for gesture in gestures:
        # Get all images for this gesture
        gesture_images = [img for img, data in annotations.items() 
                         if data['gesture'] == gesture]
        
        # Randomly select samples
        selected_images = random.sample(gesture_images, 
                                      min(samples_per_gesture, len(gesture_images)))
        
        print(f"\nVisualizing {len(selected_images)} samples for {gesture} gesture:")
        
        for img_name in selected_images:
            # Get image path and key points
            img_path = os.path.join('data/raw', img_name)
            key_points = annotations[img_name]['key_points']
            
            # Create visualization
            save_path = os.path.join(output_dir, f"vis_{img_name}")
            visualize_annotations(img_path, key_points, save_path)
            print(f"Saved visualization for {img_name}")

if __name__ == "__main__":
    main() 