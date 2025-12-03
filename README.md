# 🚜 Agricultural Laser Tracking & Targeting Simulation

This project is a **Stereo Vision-based Laser Tracking and Targeting System** simulation designed for autonomous agricultural vehicles. The system detects specific targets (weeds or crops), matches them using stereo vision, and simulates a laser locking mechanism to treat or eliminate the targets in real-time.

## 🎯 Overview

The software simulates a mechanism mounted on the back of a tractor or agricultural robot. It utilizes dual cameras (Stereo Vision) to perceive depth and coordinates, combined with **YOLOv5** for object detection. The field of view is divided into dynamic zones, where virtual lasers track and "process" targets based on a priority queue system.

### Key Features
* **Stereo Vision Processing:** Synchronized processing of Left and Right camera feeds.
* **Deep Learning Detection:** Integration with **YOLOv5** for high-accuracy object detection.
* **Stereo Matching:** Custom algorithm using Epipolar geometry and size similarity to match objects between two camera frames.
* **Multi-Zone Logic:** The operational area is divided into 4 vertical zones, each managed by an independent virtual laser.
* **Target Locking & Tracking:** Simulates a laser "burn" or "spray" action by locking onto the target's coordinates for a specific duration (1.0s).
* **Distortion Correction:** Real-time lens distortion correction using XML calibration data.

## 🛠️ Prerequisites

To run this simulation effectively, you need the following hardware and software:

### Hardware
* **GPU:** NVIDIA GPU with CUDA support (Highly recommended for YOLOv5).
* **Cameras:** 2x USB Webcams (mounted for stereo vision).

### Software & Libraries
* Python 3.8+
* PyTorch (CUDA version)
* OpenCV
* NumPy
* YOLOv5 (Must be cloned/available locally)

```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python numpy

⚙️ Configuration & Setup
⚠️ Important: Before running the script, you must update the file paths in lazers.py to match your local directory structure.
```
1. Clone the Repository
'''bash
git clone https://github.com/yourusername/repo-name.git
cd repo-name
```
2. Update Paths in lazers.py
Open the script and locate the following lines to update them with your absolute paths:

YOLOv5 Directory:
```
sys.path.append(r'C:\Path\To\Your\yolov5')
```
Model Weights (.pt file):
```
model = attempt_load(r'C:\Path\To\Your\best.pt', device='cuda')
```
Calibration Data (.xml file):
```
fs = cv2.FileStorage(r'C:\Path\To\Your\stereo_calibration_data1.xml', cv2.FILE_STORAGE_READ)
```
3. Camera Indexing
# Check your camera ports. The script uses index 0 and 2 by default. Change them if necessary:
```
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)
```
🚀 Usage
# Once configured, run the simulation:
```
python lazers.py
```
# Operational Logic
Detection: Frames are captured, undistorted, and passed to the YOLO model.

Matching: Detected objects in the Left and Right frames are matched to assign unique IDs.

Zoning: The screen is split into 4 active zones.

Action:

If a target enters a zone, the virtual laser status changes to active.

A red circle locks onto the target center.

A countdown (1.0s) begins.

Once the timer ends, the target is added to the processed list and ignored in future frames.

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the project.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.



