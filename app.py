from flask import Flask, request, render_template, redirect, url_for, session, send_file, Response
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import cv2
import json

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the pre-trained Mask R-CNN model
weights_path = "frozen_inference_graph.pb"  # Path to the pre-trained weights
config_path = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"  # Path to the configuration file
net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)

# Load COCO class names
with open("coco_classes.txt", "r") as f:
    classes = f.read().strip().split("\n")

# Generate random colors for each class
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


# def process_frame(frame):
#     """
#     Detect and visualize objects in a single frame using Mask R-CNN.
#     """
#     height, width, _ = frame.shape

#     # Prepare the image blob for input to the network
#     blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
#     net.setInput(blob)

#     # Perform forward pass to get the detections and masks
#     (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

#     # Loop through all detected objects and visualize the results
#     for i in range(boxes.shape[2]):
#         score = boxes[0, 0, i, 2]  # Get the confidence score
#         if score > 0.5:  # Only consider detections with a score higher than 0.5
#             class_id = int(boxes[0, 0, i, 1])  # Get the class id of the detected object
#             box = boxes[0, 0, i, 3:7] * np.array([width, height, width, height])
#             (x1, y1, x2, y2) = box.astype("int")  # Get the coordinates of the bounding box

#             # Draw the bounding box
#             color = [int(c) for c in colors[class_id]]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             # Display the label
#             label = f"{classes[class_id]}: {score:.2f}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # Extract and resize the mask for the object
#             mask = masks[i, class_id]
#             mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

#             # Threshold the mask and apply it to the region of interest (ROI)
#             mask = (mask > 0.5).astype("uint8") * 255  # Threshold to create a binary mask
#             roi = frame[y1:y2, x1:x2]
#             masked_roi = cv2.bitwise_and(roi, roi, mask=mask)  # Apply the mask to the ROI
#             frame[y1:y2, x1:x2] = masked_roi  # Replace the ROI with the masked version

#     return frame


def process_frame(frame):
    print("Processing frame...")
    height, width, _ = frame.shape

    # Prepare the image blob for input to the network
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass to get the detections and masks
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    # Loop through all detected objects and visualize the results
    for i in range(boxes.shape[2]):
        score = boxes[0, 0, i, 2]  # Get the confidence score
        if score > 0.5:  # Only consider detections with a score higher than 0.5
            class_id = int(boxes[0, 0, i, 1])  # Get the class id of the detected object
            box = boxes[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")  # Get the coordinates of the bounding box

            # Extract and resize the mask for the object
            mask = masks[i, class_id]
            mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            # Threshold the mask to create a binary mask
            mask = (mask > 0.5).astype("uint8")

            # Create a dark color for the mask
            color = [int(c * 0.5) for c in colors[class_id]]  # Darken the color

            # Create a colored mask overlay
            colored_mask = np.zeros_like(frame[y1:y2, x1:x2], dtype=np.uint8)
            colored_mask[mask > 0] = color  # Apply the dark color to the mask region

            # Blend the colored mask with the original frame
            alpha = 0.5  # Transparency factor (0 = fully transparent, 1 = fully opaque)
            frame[y1:y2, x1:x2] = cv2.addWeighted(colored_mask, alpha, frame[y1:y2, x1:x2], 1 - alpha, 0)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display the label with a background for better visibility
            label = f"{classes[class_id]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  # Smaller font size
            cv2.rectangle(frame, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)  # Background
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Smaller text

    print("Frame processed.")
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
    return result_path

# def detect_on_image(image_path):
#     """
#     Detect objects in an image using Mask R-CNN.
#     """
#     frame = cv2.imread(image_path)
#     result = process_frame(frame)
#     result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
#     cv2.imwrite(result_path, result)
#     return result_path


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


# Upload and process image
@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Process the image
        result_path = detect_on_image(file_path)

        # Store the filenames in the session
        session['original_image'] = filename
        session['detected_image'] = os.path.basename(result_path)

        # Debug: Print session data
        print(f"Session data - Original: {session['original_image']}, Detected: {session['detected_image']}")

        # Redirect to the result page
        return redirect(url_for('result'))

    return redirect(request.url)


# @app.route('/object-detection/', methods=['POST'])
# def apply_detection():
#     if 'image' not in request.files:
#         return redirect(request.url)

#     file = request.files['image']
#     if file.filename == '':
#         return redirect(request.url)

#     if file and allowed_file(file.filename):
#         # Save the uploaded file
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(file_path)

#         # Process the image
#         result_path = detect_on_image(file_path)

#         # Store the filenames in the session
#         session['original_image'] = filename
#         session['detected_image'] = os.path.basename(result_path)

#         # Redirect to the result page
#         return redirect(url_for('result'))

#     return redirect(request.url)


# Result page
@app.route('/result')
def result():
    if 'original_image' not in session or 'detected_image' not in session:
        return redirect(url_for('index'))

    original_image = session['original_image']
    detected_image = session['detected_image']
    return render_template('result.html', original_image=original_image, detected_image=detected_image)


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