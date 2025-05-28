import cv2
import os
import time
from datetime import datetime

class HandGestureCollector:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.cap = cv2.VideoCapture(0)
        self.gesture_label = 'default'
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def set_gesture_label(self, label):
        """Set the current gesture label for captured images."""
        self.gesture_label = label
        print(f"Current gesture label set to: {label}")
        
    def capture_frame(self):
        """Capture a single frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        return frame
    
    def save_image(self, frame):
        """Save the captured frame with timestamp and gesture label."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.gesture_label}_{timestamp}.jpg"
        filepath = os.path.join(self.data_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved image: {filename}")
        
    def run(self):
        """Main loop for capturing images."""
        print("Starting hand gesture collection...")
        print("Press 'c' to capture an image")
        print("Press '1-5' to set gesture label (1: open, 2: closed, 3: pointing, 4: pinching, 5: waving)")
        print("Press 'q' to quit")
        
        while True:
            frame = self.capture_frame()
            if frame is None:
                continue
                
            # Display the frame
            cv2.imshow('Hand Gesture Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.save_image(frame)
            elif key == ord('1'):
                self.set_gesture_label('open')
            elif key == ord('2'):
                self.set_gesture_label('closed')
            elif key == ord('3'):
                self.set_gesture_label('pointing')
            elif key == ord('4'):
                self.set_gesture_label('pinching')
            elif key == ord('5'):
                self.set_gesture_label('waving')
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = HandGestureCollector()
    collector.run() 