# Hand Gesture Mouse Control

A project to control your computer mouse using hand gestures detected via a webcam and custom machine learning models.

---

## Project Phases & TODO List

### Phase 1: Setup & Data Collection ✅
- [x] Set up Python virtual environment
- [x] Install dependencies (OpenCV, PyAutoGUI, TensorFlow/PyTorch, etc.)
- [x] Create scripts to capture and label hand images/videos
- [x] Collect and annotate dataset for hand landmarks

### Phase 2: Model Development
- [ ] Preprocess collected data (resize, normalize, augment, etc.)
- [ ] Design and implement a custom hand landmark detection model
- [ ] Train the model and evaluate performance
- [ ] Save trained model checkpoints

### Phase 3: Inference & Gesture Recognition
- [ ] Write inference script to run model on webcam feed
- [ ] Implement gesture recognition logic (e.g., swipe, pinch, etc.)

### Phase 4: Mouse Control Integration
- [ ] Integrate PyAutoGUI to control mouse based on recognized gestures
- [ ] Test and refine gesture-to-mouse mapping

### Phase 5: Testing & Documentation
- [ ] Write unit and integration tests
- [ ] Document code and usage in README.md
- [ ] Test on both Mac and Windows

### Phase 6: Version Control & Collaboration ✅
- [x] Initialize git repository
- [x] Create .gitignore (ignore data, models, etc.)
- [x] Push to GitHub

---

## Project Structure (Suggested)

```
hand-gesture-mouse-control/
│
├── data/                   # Training images, videos, and labels
├── models/                 # Saved models and checkpoints
├── notebooks/              # Jupyter notebooks for prototyping
├── src/                    # Source code
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── .gitignore              # Files/folders to ignore in git
```

---

## Getting Started

1. Clone the repository:
   ```sh
   git clone git@github.com:kingelvyn/hand-gesture-mouse.git
   cd hand-gesture-mouse
   ```
2. Set up your Python environment and install dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
