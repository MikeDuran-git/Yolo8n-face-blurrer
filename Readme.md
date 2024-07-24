# Face Detection Program Report
## Overview

This report details a Python script designed to detect and blur faces in real-time using a webcam feed. The script leverages the YOLOv8n-face model, a variant of the YOLO machine learning models optimized for face detection. This document serves as a guide for executing the program and understanding its core functionalities.

# Program Components and Functionality

## Model Loading
The script initiates by loading the YOLOv8n-face model from the file yolov8n-face.pt.
This model is adept at detecting faces within various environments, ensuring robust performance across different settings.

## Video Stream Initialization
A video stream is established, typically utilizing the device's default camera.
The script confirms the camera's operational status and retrieves essential metrics like dimensions and frame rate.

## Real-time Face Detection and Blurring
The program processes the video feed frame by frame, utilizing the YOLOv8n model to identify faces.
Detected faces are blurred based on a specified blur ratio, ensuring privacy.

## Display and User Control
Processed frames, with faces blurred, are displayed in real-time.
Users can terminate the program by pressing `q` or triggering a `KeyboardInterrupt (usually by pressing Ctrl+C)`.

# Execution Instructions
1. **Setting Up a Python Environment:**

- Create a new virtual environment to manage dependencies: python -m venv face-detection-env.
- Activate the virtual environment:
  - On Windows: 
    ```cmd
    face-detection-env\Scripts\activate
    ```
  - On macOS/Linux: 
    ```bash
    source face-detection-env/bin/activate
    ```
2. **Installing Dependencies:**

- Install necessary Python packages: 
  ```bash
  pip install ultralytics opencv-python
  ``` 
1. **Running the Script:**

- Ensure you're in the directory containing the script.
- Execute the script using Python: 
    
    ```bash 
    python yolov8n-face-detection.py
    ```

4. **Program Interaction:**
- The video feed will appear in a new window, showing the detected and blurred faces.
- To exit, press `q` or use a `KeyboardInterrupt (Ctrl+C)`.

# Understanding the YOLOv8n-face Model

The YOLOv8n-face model is an advanced neural network trained for the specific task of face detection. It belongs to the YOLO (You Only Look Once) family, known for its efficiency in processing images in real-time. This model analyzes images in a single evaluation, making it highly efficient for applications requiring immediate output, like live video processing. Its training on diverse datasets ensures reliable face detection under various conditions.

# Conclusion:
This Python script demonstrates a practical application of AI in privacy-oriented tasks, utilizing state-of-the-art technology to detect and anonymize faces in real-time video feeds. By following the outlined steps, you can effectively execute the program and benefit from its capabilities.

# Author:

Author: Mike Duran
Company: Actimage GmbH
Adress:  Hafenstraße 3, 77694 Kehl
Contact: +49 7851 899730

# Source:
Yolo8: https://github.com/ultralytics/ultralytics
Licence: AGPL-3.0 licence


