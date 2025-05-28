import cv2
import os
import json
import time
from datetime import datetime
import numpy as np

class DataCollector:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.gestures = ['open', 'closed', 'pointing', 'pinching', 'waving']
        self.current_gesture = 0
        self.counter = 0
        self.required_samples = 200  # samples per gesture
        self.cap = cv2.VideoCapture(0)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing annotations if any
        self.annotations_file = "data/processed/hand_features.json"
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
    
    def get_next_filename(self):
        """Generate unique filename for the image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.gestures[self.current_gesture]}_{timestamp}_{self.counter}.jpg"
    
    def save_image(self, frame):
        """Save the image and update annotations."""
        filename = self.get_next_filename()
        filepath = os.path.join(self.data_dir, filename)
        cv2.imwrite(filepath, frame)
        
        # Update annotations
        self.annotations[filename] = {
            'gesture': self.gestures[self.current_gesture],
            'timestamp': datetime.now().isoformat(),
            'key_points': []  # Will be filled during annotation
        }
        
        # Save annotations
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        
        self.counter += 1
        return filename
    
    def display_info(self, frame):
        """Display collection information on the frame."""
        # Add gesture name
        cv2.putText(frame, f"Gesture: {self.gestures[self.current_gesture]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add counter
        cv2.putText(frame, f"Count: {self.counter}/{self.required_samples}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Press SPACE to capture", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press N for next gesture", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        print("Starting data collection...")
        print("Instructions:")
        print("- Press SPACE to capture an image")
        print("- Press N to move to next gesture")
        print("- Press Q to quit")
        print(f"\nCollecting data for gesture: {self.gestures[self.current_gesture]}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Add information overlay
            frame = self.display_info(frame)
            
            # Show the frame
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space
                if self.counter < self.required_samples:
                    filename = self.save_image(frame)
                    print(f"Saved {filename}")
                else:
                    print(f"Completed {self.required_samples} samples for {self.gestures[self.current_gesture]}")
            elif key == ord('n'):  # Next gesture
                if self.counter >= self.required_samples:
                    self.current_gesture = (self.current_gesture + 1) % len(self.gestures)
                    self.counter = 0
                    print(f"\nCollecting data for gesture: {self.gestures[self.current_gesture]}")
                else:
                    print(f"Please complete {self.required_samples} samples for current gesture")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\nData Collection Summary:")
        for gesture in self.gestures:
            count = sum(1 for v in self.annotations.values() if v['gesture'] == gesture)
            print(f"{gesture}: {count} samples")
        
        print("\nTotal samples collected:", len(self.annotations))

if __name__ == "__main__":
    collector = DataCollector()
    collector.run() 