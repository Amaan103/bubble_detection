# ABEx Bubble Detector

## Features
- Real-time 4-tier bubble classification (NO/LOW/MEDIUM/HIGH)
- Full frame analysis
- Multi-sensor fusion: Area + Motion + Edge density + LBP texture
- Hysteresis tracking for stable state transitions
- **Jetson Nano optimized** (15-25 FPS target)
- Real-time CSV logging for analysis
- USB Webcam ready

## Requirements
- Python 3.8+
- USB Webcam
- OpenCV 4.5+

## Installation
```bash

# Install dependencies
pip3 install opencv-python==4.8.1.78 numpy

# Clone repository
cd ~
git clone https://github.com/Amaan103/bubble_detection

# Test camera access
python3 bubble_detector.py --test-camera

```

## Running The System
```bash
# Live detection
python3 bubble_detector.py
# Press 'q' to quit

# Headless mode (no display)
python3 bubble_detector.py --headless

# Analyze recorded video
python3 bubble_detector.py --video test_sample.mp4
```

## ROS Package Guide
```bash
# 1. Create package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python abex_bubble_detector \
    --dependencies rclpy sensor_msgs std_msgs cv_bridge

# 2. Build & run
cd ~/ros2_ws
colcon build --packages-select abex_bubble_detector
source install/setup.bash
ros2 run abex_bubble_detector bubble_detector
```

## System Architecture
```bash
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│   USB Webcam │───▶│  Processing     │───▶│   CSV Log    │
│  Full Frame  │    │   Pipeline      │    │  Analysis    │
└──────────────┘    │                 │    └──────────────┘
                     │ -  CLAHE Enhance │
                     │ -  MOG2 Foreground│
                     │ -  Optical Flow  │
                     │ -  Canny Edges   │
                     │ -  Fusion Score  │
                     └─────────────────┘
```

## Adjustable Parameters
```bash
# Command line tuning
python3 bubble_detector.py \
    --thresh-low 17.5 \
    --thresh-med 22.5 \
    --thresh-high 27.5 \
    --motion-scale 3.0
```

## Detection Categories
```bash
| Category  | Intensity Range | ABEx Mission Status      |
| --------- | --------------- | ------------------------ |
| NO BUBBLE | <14.5%          | Stable fluid sample      |
| LOW       | 14.5-18.5%      | Minor gas evolution      |
| MEDIUM    | 18.5-22.5%      | Active chemical reaction |
| HIGH      | >27.5%          | Strong gas production    |
```

## Troubleshooting
```bash
 Camera not detected:
ls -l /dev/video*
sudo usermod -a -G video $USER  # Log out/in


 False positives:
Increase HYSTERESIS_BUFFER=1.0
Check uniform LED lighting

 No detection:
Verify sample illumination
Test with --video known_bubble.mp4

```

