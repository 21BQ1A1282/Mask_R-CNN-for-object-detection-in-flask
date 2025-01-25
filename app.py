from flask import Flask, request, render_template, redirect, url_for, session, send_file, Response
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import json
import random
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
MODEL_FOLDER = 'static/models/'
DATASET_FOLDER = 'static/datasets/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Load the pre-trained Mask R-CNN model
weights_path = os.path.join(MODEL_FOLDER, "frozen_inference_graph.pb")  # Path to the pre-trained weights
config_path = os.path.join(MODEL_FOLDER, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")  # Path to the configuration file
net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)

# Load COCO class names
with open("coco_classes.txt", "r") as f:
    classes = f.read().strip().split("\n")

# Generate random colors for each class
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")


def allowed_file(filename, extensions=None):
    """
    Check if the uploaded file has an allowed extension.
    """
    if extensions is None:
        extensions = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions





def draw_dashed_rectangle(image, top_left, bottom_right, color, thickness=2, dash_length=10):
    """
    Draw a dashed rectangle on the image.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw top line
    for x in range(x1, x2, dash_length * 2):
        cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), color, thickness)

    # Draw bottom line
    for x in range(x1, x2, dash_length * 2):
        cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

    # Draw left line
    for y in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)

    # Draw right line
    for y in range(y1, y2, dash_length * 2):
        cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def process_frame(frame):
    """
    Detect and visualize objects in a single frame using Mask R-CNN.
    """
    height, width, _ = frame.shape

    # Prepare the image blob for input to the network
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass to get the detections and masks
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    # Generate unique colors for each object
    object_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Pink
        (128, 0, 128),  # Violet
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
    ]

    # Loop through all detected objects and visualize the results
    for i in range(boxes.shape[2]):
        score = boxes[0, 0, i, 2]  # Get the confidence score
        if score > 0.5:  # Only consider detections with a score higher than 0.5
            class_id = int(boxes[0, 0, i, 1])  # Get the class id of the detected object
            box = boxes[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")  # Get the coordinates of the bounding box

            # Extract and resize the mask for the object
            mask = masks[i, class_id]
            mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

            # Apply Gaussian blur to smooth the mask
            mask = cv2.GaussianBlur(mask, (7, 7), 0)

            # Threshold the mask to create a binary mask
            mask = (mask > 0.5).astype("uint8") * 255  # Scale to 0-255

            # Refine the mask using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise

            # Find contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a blank mask for the refined shape
            refined_mask = np.zeros_like(mask)

            # Draw the largest contour (main object) on the refined mask
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Smooth the contour using contour approximation
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Draw the smoothed contour on the refined mask
            refined_mask = np.zeros_like(mask)
            cv2.drawContours(refined_mask, [smoothed_contour], -1, 255, thickness=cv2.FILLED)

            # Apply the refined mask to the original frame (inside the bounding box only)
            color = object_colors[i % len(object_colors)]  # Cycle through unique colors
            colored_mask = np.zeros_like(frame[y1:y2, x1:x2], dtype=np.uint8)
            colored_mask[refined_mask > 0] = color  # Apply the color to the refined mask region

            # Blend the colored mask with the original frame using alpha blending (inside the bounding box only)
            alpha = 0.5  # Adjust transparency for better blending
            frame[y1:y2, x1:x2] = cv2.addWeighted(colored_mask, alpha, frame[y1:y2, x1:x2], 1 - alpha, 0)

            # Draw the bounding box with dashed lines
            draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, thickness=2, dash_length=10)

            # Draw the edge shape (contour) of the object (inside the bounding box only)
            for contour in contours:
                # Scale the contour to the original image coordinates
                contour = contour + np.array([[x1, y1]])
                # Draw the contour on the frame
                cv2.drawContours(frame, [contour], -1, color, thickness=1)  # Contour thickness

            # Display the label with a background for better visibility
            label = f"{classes[class_id]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  # Smaller font size
            cv2.rectangle(frame, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)  # Background
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Smaller text

    return frame


def detect_on_image(image_path):
    """
    Detect objects in an image using Mask R-CNN.
    """
    frame = cv2.imread(image_path)
    result = process_frame(frame)

    # Save the processed image
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, result)
    print("Original file path:", image_path)
    print("Processed file path:", result_path)
    return result_path


def detect_on_video(video_path):
    """
    Detect objects in a video using Mask R-CNN.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the output video path
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(video_path))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame using the Mask R-CNN model
        processed_frame = process_frame(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    return result_path




# Home route
@app.route('/')
def home():
    return render_template('first.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['login-username']
        password = request.form['login-password']
        users = load_users()
        user = next((user for user in users if user['username'] == username and user['password'] == password), None)
        if user:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))  # Redirect to index.html
        else:
            return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')


# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['signup-username']
        email = request.form['signup-email']
        password = request.form['signup-password']
        users = load_users()
        if any(user['username'] == username for user in users):
            return render_template('signup.html', error="Username already exists.")
        users.append({'username': username, 'email': email, 'password': password})
        save_users(users)
        return redirect(url_for('login'))
    return render_template('signup.html')


# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


# Object detection route

@app.route('/index')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  # Redirect to login if not logged in

    # Pass the processed file path to the template
    processed_file = session.get('processed_file', None)
    original_file = session.get('original_file', None)
    return render_template('index.html', original_file=original_file, processed_file=processed_file)


# Upload and process image or video
@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Check if the file is an image or video
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Process the image
            result_path = detect_on_image(file_path)
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Process the video
            result_path = detect_on_video(file_path)
        else:
            return redirect(request.url)

        # Store the filenames in the session
        session['original_file'] = filename
        session['processed_file'] = os.path.basename(result_path)

        # Redirect to the index page with the result
        return redirect(url_for('index'))

    return redirect(request.url)


@app.route('/reset', methods=['POST'])
def reset():
    session.pop('original_file', None)
    session.pop('processed_file', None)
    return redirect(url_for('index'))

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video')
def video():
    return render_template('video.html')

def gen_frames():
    cap = cv2.VideoCapture(0)  # Open the default camera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for better performance
        frame = cv2.resize(frame, (640, 480))

        # Process the frame using the Mask R-CNN model
        processed_frame = process_frame(frame)

        # Encode the processed frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Admin Login
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':  # Replace with secure authentication
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin/login.html', error="Invalid username or password.")
    return render_template('admin/login.html')


# Admin Logout
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))


# Admin Dashboard
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('admin/dashboard.html')


# Model Management
@app.route('/admin/model-management', methods=['GET', 'POST'])
def model_management():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        if 'model' in request.files:
            model_file = request.files['model']
            if model_file and allowed_file(model_file.filename, {'pb', 'pbtxt'}):
                filename = secure_filename(model_file.filename)
                model_file.save(os.path.join(MODEL_FOLDER, filename))
                return redirect(url_for('model_management'))

    # List existing models
    models = os.listdir(MODEL_FOLDER)
    return render_template('admin/model_management.html', models=models)


# Dataset Management
@app.route('/admin/dataset-management', methods=['GET', 'POST'])
def dataset_management():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        if 'dataset' in request.files:
            dataset_file = request.files['dataset']
            if dataset_file and allowed_file(dataset_file.filename, {'zip', 'tar', 'gz'}):
                filename = secure_filename(dataset_file.filename)
                dataset_file.save(os.path.join(DATASET_FOLDER, filename))
                return redirect(url_for('dataset_management'))

    # List existing datasets
    datasets = os.listdir(DATASET_FOLDER)
    return render_template('admin/dataset_management.html', datasets=datasets)


# Performance Monitoring
@app.route('/admin/performance-monitoring')
def performance_monitoring():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    # Example ground truth and detection data
    ground_truths = [
        {'bbox': [10, 10, 50, 50], 'class_id': 1},  # [x1, y1, x2, y2]
        {'bbox': [60, 60, 100, 100], 'class_id': 2}
    ]

    detections = [
        {'bbox': [12, 12, 52, 52], 'class_id': 1, 'score': 0.95},  # [x1, y1, x2, y2]
        {'bbox': [65, 65, 105, 105], 'class_id': 2, 'score': 0.90}
    ]

    # Compute mAP and IoU
    map_score, iou_score = compute_map_iou_custom(ground_truths, detections)

    # Compute processing speed (without using the webcam)
    fps = compute_processing_speed(use_webcam=False)

    # Load user activity logs
    user_activity = []
    if os.path.exists('user_activity_log.json'):
        with open('user_activity_log.json', 'r') as log_file:
            for line in log_file:
                user_activity.append(json.loads(line))

    # Prepare performance metrics
    performance_metrics = {
        'mAP': map_score,
        'IoU': iou_score,
        'processing_speed': f'{fps:.2f} fps',
        'user_activity': user_activity
    }

    return render_template('admin/performance_monitoring.html', metrics=performance_metrics)

def log_user_activity(user, action):
    """
    Log user activity to a file or database.
    """
    log_entry = {
        'user': user,
        'action': action,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('user_activity_log.json', 'a') as log_file:
        json.dump(log_entry, log_file)
        log_file.write('\n')

def compute_map_iou(ground_truth_file, detection_results_file):
    """
    Compute mAP and IoU using COCO evaluation tools.
    """
    coco_gt = COCO(ground_truth_file)  # Load ground truth annotations
    coco_dt = coco_gt.loadRes(detection_results_file)  # Load detection results

    # Initialize COCO evaluation object
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mAP and IoU
    map_score = coco_eval.stats[0]  # mAP @ IoU=0.50:0.95
    iou_score = coco_eval.stats[1]  # mAP @ IoU=0.50

    return map_score, iou_score


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Bounding boxes are in the format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def compute_ap(ground_truths, detections, iou_threshold=0.5):
    """
    Compute Average Precision (AP) for a single class.
    """
    # Sort detections by confidence score (descending order)
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    # Initialize variables
    true_positives = np.zeros(len(detections))
    false_positives = np.zeros(len(detections))
    matched_gt = set()

    # Match detections to ground truth boxes
    for i, det in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue  # Skip already matched ground truth boxes

            iou = compute_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if the best IoU exceeds the threshold
        if best_iou >= iou_threshold:
            true_positives[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives[i] = 1

    # Compute precision and recall
    cum_true_positives = np.cumsum(true_positives)
    cum_false_positives = np.cumsum(false_positives)
    precision = cum_true_positives / (cum_true_positives + cum_false_positives)
    recall = cum_true_positives / len(ground_truths)

    # Compute AP as the area under the Precision-Recall curve
    ap = 0
    for r in np.arange(0, 1.1, 0.1):
        precisions_at_recall = precision[recall >= r]
        if len(precisions_at_recall) > 0:
            ap += np.max(precisions_at_recall)
    ap /= 11  # Average over 11 recall levels

    return ap


def compute_map_iou_custom(ground_truths, detections, iou_threshold=0.5):
    """
    Compute mAP and IoU using custom implementation.
    """
    # Group ground truths and detections by class
    class_to_ground_truths = {}
    class_to_detections = {}

    for gt in ground_truths:
        class_id = gt['class_id']
        if class_id not in class_to_ground_truths:
            class_to_ground_truths[class_id] = []
        class_to_ground_truths[class_id].append(gt)

    for det in detections:
        class_id = det['class_id']
        if class_id not in class_to_detections:
            class_to_detections[class_id] = []
        class_to_detections[class_id].append(det)

    # Compute AP for each class
    aps = []
    ious = []

    for class_id in class_to_ground_truths:
        if class_id in class_to_detections:
            ap = compute_ap(class_to_ground_truths[class_id], class_to_detections[class_id], iou_threshold)
            aps.append(ap)

            # Compute IoU for matched detections
            for det in class_to_detections[class_id]:
                for gt in class_to_ground_truths[class_id]:
                    iou = compute_iou(det['bbox'], gt['bbox'])
                    if iou >= iou_threshold:
                        ious.append(iou)

    # Compute mAP and mean IoU
    map_score = np.mean(aps) if aps else 0
    iou_score = np.mean(ious) if ious else 0

    return map_score, iou_score

def compute_processing_speed(use_webcam=False):
    """
    Compute the average processing speed (FPS) of the model.
    If use_webcam is False, return a static value.
    """
    if not use_webcam:
        return 30.0  # Example static value

    # Use webcam to compute FPS
    cap = cv2.VideoCapture(0)  # Use a video file or webcam
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        process_frame(frame)
        frame_count += 1

        # Stop after processing 100 frames
        if frame_count >= 100:
            break

    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    cap.release()
    return fps

# Helper functions for user management
def load_users():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as file:
            return json.load(file)
    return []


def save_users(users):
    with open('users.json', 'w') as file:
        json.dump(users, file)


# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)