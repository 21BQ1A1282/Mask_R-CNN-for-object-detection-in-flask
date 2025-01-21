
# Mask R-CNN for Object Detection Application with Flask

This is a Flask-based web application that performs object detection on images using the Mask R-CNN model. Users can upload an image, and the application will detect and highlight objects in the image using bounding boxes and masks.

## Features
- User authentication (login/signup).
- Upload images for object detection.
- Display original and processed images with detected objects.
- Real-time object detection using a pre-trained Mask R-CNN model.


## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/21BQ1A1282/Mask_R-CNN-for-object-detection-in-flask.git
   cd object-detection-app
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained Mask R-CNN model files:
   - frozen_inference_graph.pb
   - mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
   - coco_classes.txt
   Place these files in the root directory of the project.

## Running the Application
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to:
   ```bash
   http://localhost:8000
   ```
3. Use the application:
   - Sign up or log in to access the object detection feature.
   - Upload an image and view the detected objects.
