from flask import Flask, request, render_template, redirect, url_for, session, send_file, Response
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import json
import random
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

# Define object_colors globally
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

def process_frame(frame, class_name=None):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    
    scores = []  # Collect confidence scores here

    # Loop through all detected objects
    for i in range(boxes.shape[2]):
        score = boxes[0, 0, i, 2]
        if score > 0.5:  # Only consider detections with score > 0.5
            class_id = int(boxes[0, 0, i, 1])
            class_label = classes[class_id]
            if class_name and class_label.lower() != class_name.lower():
                continue  # Skip if class doesn't match filter
            scores.append(score)

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
            label = f"{class_label}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  # Smaller font size
            cv2.rectangle(frame, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)  # Background
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Smaller text

    return frame, scores

def detect_on_image(image_path, class_name=None):
    frame = cv2.imread(image_path)
    processed_frame, scores = process_frame(frame, class_name)
    # Calculate average confidence (convert to percentage)
    avg_confidence = (sum(scores) / len(scores)) * 100 if scores else 0
    # Save the result
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, processed_frame)
    # Store average confidence in session
    session['accuracy'] = round(avg_confidence, 2)
    return result_path


def detect_on_video(video_path, class_name=None):
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
    all_scores = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, scores = process_frame(frame, class_name)
        all_scores.extend(scores)
        out.write(processed_frame)
    # Calculate average confidence
    avg_confidence = (sum(all_scores) / len(all_scores)) * 100 if all_scores else 0
    session['accuracy'] = round(avg_confidence, 2)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    return result_path




def detect_image_in_image(source_path, target_path):
    """
    Detect if target image appears within source image
    Returns confidence score and processed image with detection highlighted
    """
    # Read images
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    # Convert images to grayscale
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(target_gray, None)
    kp2, des2 = sift.detectAndCompute(source_gray, None)
    
    # Initialize parameters for matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0, source_img
    
    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Calculate confidence score
    confidence_score = len(good_matches) / len(kp1) if len(kp1) > 0 else 0
    
    # If we have enough good matches, find the object
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            # Get dimensions of target image
            h, w = target_gray.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # Transform corners into source image space
            dst = cv2.perspectiveTransform(pts, H)
            
            # Draw detected object
            result = cv2.polylines(source_img, [np.int32(dst)], True, (0, 255, 0), 3)
            
            # Add confidence score to image
            cv2.putText(result, f'Match Score: {confidence_score:.2%}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return confidence_score, result
    
    return confidence_score, source_img


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
    return render_template('index.html')


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

        # Process the file based on its type
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            result_path = detect_on_image(file_path)
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            result_path = detect_on_video(file_path)
        else:
            return "Unsupported file format", 400

        # Save the result file
        result_filename = 'detection_' + filename
        result_path_final = os.path.join(RESULT_FOLDER, result_filename)
        os.rename(result_path, result_path_final)  # Move or rename the result file

        # Store filenames and accuracy in session
        session['original_file'] = filename
        session['processed_file'] = result_filename
        session['accuracy'] = session.get('accuracy', 0)  # Assuming accuracy is set elsewhere

        # Redirect to the result page
        return redirect(url_for('detection_result'))

    return redirect(request.url)


@app.route('/filter_detection', methods=['POST'])
def filter_detection():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    class_name = request.form['class_name']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Process the file based on its type
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            result_path = detect_on_image(file_path, class_name)
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            result_path = detect_on_video(file_path, class_name)
        else:
            return "Unsupported file format", 400

        # Save the result file
        result_filename = 'filtered_' + filename
        result_path_final = os.path.join(RESULT_FOLDER, result_filename)
        os.rename(result_path, result_path_final)  # Move or rename the result file

        # Store filenames and accuracy in session
        session['original_file'] = filename
        session['processed_file'] = result_filename
        session['accuracy'] = session.get('accuracy', 0)  # Assuming accuracy is set elsewhere

        # Redirect to the result page
        return redirect(url_for('detection_result'))

    return redirect(request.url)


# Add new route for image matching
@app.route('/match-images', methods=['POST'])
def match_images():
    if 'source_image' not in request.files or 'target_image' not in request.files:
        return redirect(request.url)
        
    source_file = request.files['source_image']
    target_file = request.files['target_image']
    
    if source_file.filename == '' or target_file.filename == '':
        return redirect(request.url)
        
    if (source_file and allowed_file(source_file.filename) and 
        target_file and allowed_file(target_file.filename)):
        
        # Save uploaded files
        source_filename = secure_filename(source_file.filename)
        target_filename = secure_filename(target_file.filename)
        
        source_path = os.path.join(UPLOAD_FOLDER, source_filename)
        target_path = os.path.join(UPLOAD_FOLDER, target_filename)
        
        source_file.save(source_path)
        target_file.save(target_path)
        
        # Process images
        confidence_score, result_img = detect_image_in_image(source_path, target_path)
        
        # Save result
        result_filename = 'match_' + source_filename
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result_img)
        
        # Store results in session
        session['match_result'] = result_filename
        session['match_confidence'] = round(confidence_score * 100, 2)
        
        return redirect(url_for('match_result'))
        
    return redirect(url_for('index'))

# Add route for showing match results
@app.route('/match-result')
def match_result():
    if 'match_result' not in session:
        return redirect(url_for('index'))
        
    return render_template('match_result.html',
                         result_file=session['match_result'],
                         confidence=session['match_confidence'])





@app.route('/detection-result')
def detection_result():
    if 'original_file' not in session or 'processed_file' not in session:
        return redirect(url_for('index'))
    
    # Retrieve accuracy from session, default to 0 if not found
    accuracy = session.get('accuracy', 0)
    
    return render_template('detection_result.html', 
                         original_file=session['original_file'],
                         processed_file=session['processed_file'],
                         accuracy=accuracy)


# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return render_template('video.html')

def gen_frames(class_name=None):  # Allow class_name to be optional
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Failed to grab frame.")
            break

        # Resize the frame for better performance
        frame = cv2.resize(frame, (640, 480))

        # Process the frame using the Mask R-CNN model
        processed_frame, _ = process_frame(frame, class_name)

        # Ensure processed_frame is a valid NumPy array
        if processed_frame is None or not isinstance(processed_frame, np.ndarray):
            print("Error: Processed frame is invalid.")
            continue

        # Convert to uint8 if necessary
        if processed_frame.dtype != np.uint8:
            processed_frame = processed_frame.astype(np.uint8)

        # Encode the processed frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Error: Could not encode frame.")
            continue

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
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8000, debug=True)