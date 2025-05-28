import cv2
import os
import json
import numpy as np
from pathlib import Path

class HandLandmarkAnnotator:
    def __init__(self, data_dir='data/raw', output_dir='data/processed'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Skin color ranges in HSV
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
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
        """Detect hand in the image using improved color-based segmentation."""
        # Preprocess image
        processed, original_size = self.preprocess_image(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
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
            return None, None, mask, processed
        
        # Get the largest contour (should be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter out small detections
        min_size = 50
        if w < min_size or h < min_size:
            return None, None, mask, processed
            
        # Ensure the contour is large enough
        if cv2.contourArea(largest_contour) < 1000:
            return None, None, mask, processed
            
        return (x, y, w, h), largest_contour, mask, processed
        
    def extract_hand_features(self, image, hand_rect, contour):
        """Extract improved hand features from the detected region."""
        x, y, w, h = hand_rect
        
        # Get convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        # Extract key points
        key_points = []
        if defects is not None:
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
        """Process a single image and detect hand features."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
            
        # Detect hand
        hand_rect, contour, mask, processed = self.detect_hand(image)
        
        if hand_rect is None:
            print(f"No hand detected in: {image_path}")
            return None
            
        # Extract features
        key_points = self.extract_hand_features(processed, hand_rect, contour)
        
        if not key_points:
            print(f"Could not extract features from: {image_path}")
            return None
            
        # Create visualization
        annotated_image = processed.copy()
        x, y, w, h = hand_rect
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw contour
        cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)
        
        # Draw key points
        for point in key_points:
            px = int(point['x'] * processed.shape[1])
            py = int(point['y'] * processed.shape[0])
            cv2.circle(annotated_image, (px, py), 5, (0, 0, 255), -1)
            
        # Create debug visualization
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug_image = np.hstack([processed, mask_bgr])
        
        return annotated_image, debug_image, key_points
        
    def process_all_images(self):
        """Process all images in the data directory."""
        # Get all image files
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Create a dictionary to store all annotations
        annotations = {}
        
        for image_file in image_files:
            print(f"Processing {image_file}...")
            image_path = os.path.join(self.data_dir, image_file)
            
            # Process image
            result = self.process_image(image_path)
            if result is None:
                continue
                
            annotated_image, debug_image, key_points = result
            
            # Save annotated image
            output_image_path = os.path.join(self.output_dir, f"annotated_{image_file}")
            cv2.imwrite(output_image_path, annotated_image)
            
            # Save debug image
            debug_image_path = os.path.join(self.output_dir, f"debug_{image_file}")
            cv2.imwrite(debug_image_path, debug_image)
            
            # Store key points
            annotations[image_file] = {
                'key_points': key_points,
                'gesture': image_file.split('_')[0]  # Extract gesture label from filename
            }
            
        # Save annotations to JSON
        with open(os.path.join(self.output_dir, 'hand_features.json'), 'w') as f:
            json.dump(annotations, f, indent=2)
            
        print(f"\nProcessed {len(annotations)} images")
        print(f"Annotations saved to: {os.path.join(self.output_dir, 'hand_features.json')}")
        print(f"Annotated images saved to: {self.output_dir}")

if __name__ == "__main__":
    annotator = HandLandmarkAnnotator()
    annotator.process_all_images() 