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
