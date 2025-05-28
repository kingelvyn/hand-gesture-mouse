import cv2
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

class HandLandmarkAnnotator:
    def __init__(self, data_dir='data/raw', output_dir='data/processed'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Skin color ranges in HSV
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Load annotations
        self.annotations_file = os.path.join(output_dir, 'hand_features.json')
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
    
    def preprocess_image(self, image):
        """Preprocess image for better hand detection."""
        # Resize image for faster processing
        height, width = image.shape[:2]
        max_dim = 640
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        return blurred, image.shape[:2]  # Return both processed image and original size
    
    def detect_hand(self, image):
        """Detect hand in the image using color-based segmentation."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Get the largest contour (should be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter out small detections
        min_size = 50
        if w < min_size or h < min_size:
            return None, None
        
        # Ensure the contour is large enough
        if cv2.contourArea(largest_contour) < 1000:
            return None, None
        
        return (x, y, w, h), largest_contour
    
    def extract_features(self, image, hand_rect, contour):
        """Extract hand features from the detected region."""
        x, y, w, h = hand_rect
        
        # Get convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return None
        
        # Extract key points
        key_points = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Filter defects based on distance
            if d > 10000:  # Adjust this threshold as needed
                # Convert to relative coordinates
                key_points.append({
                    'x': far[0] / image.shape[1],
                    'y': far[1] / image.shape[0],
                    'depth': d / 256.0  # Normalize depth
                })
        
        return key_points
    
    def process_image(self, image_path):
        """Process a single image and extract hand features."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        # Preprocess image
        processed_image, original_size = self.preprocess_image(image)
        
        # Detect hand
        hand_rect, contour = self.detect_hand(processed_image)
        if hand_rect is None:
            print(f"No hand detected in: {image_path}")
            return None
        
        # Extract features
        key_points = self.extract_features(processed_image, hand_rect, contour)
        if key_points is None:
            print(f"Failed to extract features from: {image_path}")
            return None
        
        return key_points
    
    def annotate_all(self):
        """Process all images in the data directory."""
        print("Starting annotation process...")
        
        # Get list of all images
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)
        
        print(f"Found {total_images} images to process")
        
        # Process each image with progress bar
        for image_file in tqdm(image_files, desc="Annotating images"):
            image_path = os.path.join(self.data_dir, image_file)
            
            # Skip if already annotated
            if image_file in self.annotations and self.annotations[image_file].get('key_points'):
                continue
            
            # Process image
            key_points = self.process_image(image_path)
            if key_points is not None:
                # Update annotations
                if image_file not in self.annotations:
                    self.annotations[image_file] = {
                        'gesture': image_file.split('_')[0],  # Get gesture from filename
                        'timestamp': self.annotations.get(image_file, {}).get('timestamp', ''),
                        'key_points': key_points
                    }
                else:
                    self.annotations[image_file]['key_points'] = key_points
                
                # Save annotations periodically
                if len(self.annotations) % 10 == 0:
                    self.save_annotations()
        
        # Save final annotations
        self.save_annotations()
        
        # Print summary
        print("\nAnnotation Summary:")
        gestures = {}
        for data in self.annotations.values():
            gesture = data['gesture']
            gestures[gesture] = gestures.get(gesture, 0) + 1
        
        for gesture, count in gestures.items():
            print(f"{gesture}: {count} annotated samples")
        
        print(f"\nTotal annotated samples: {len(self.annotations)}")
    
    def save_annotations(self):
        """Save annotations to file."""
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)

if __name__ == "__main__":
    annotator = HandLandmarkAnnotator()
    annotator.annotate_all() 