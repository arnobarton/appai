from flask import Flask, render_template, request, jsonify
import cv2
import os
import torch
import numpy as np
import base64

app = Flask(__name__)

# Load YOLOv5 model# Define the model path
model_path = os.path.join('models', 'best.pt')  # Adjust path as needed

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Ensure it's in the correct location.")

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the frame from the request
        data = request.json['image']
        _, encoded = data.split(',', 1)  # Split out the base64 data
        decoded = base64.b64decode(encoded)  # Decode the base64 string
        nparr = np.frombuffer(decoded, np.uint8)  # Convert to numpy array
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image

        # Run YOLO inference
        results = model(frame)
        detected_objects = results.pandas().xyxy[0].to_dict('records')  # Extract detections

        # Debugging: Log the detected objects
        print("Detected objects:", detected_objects)

        # Extract detected classes
        detected_classes = [obj['name'] for obj in detected_objects]

        # Determine detection status
        if "Smoke" in detected_classes and "fire" in detected_classes:
            detection = "Fire and Smoke Detected!"
        elif "Smoke" in detected_classes:
            detection = "Smoke Detected!"
        elif "fire" in detected_classes:
            detection = "Fire Detected!"
        else:
            detection = "No Fire or Smoke Detected"

        return jsonify({
            'status': 'success',
            'detection': detection,
            'details': detected_objects
        })

    except Exception as e:
        print(f"Error in /detect endpoint: {e}")  # Debugging
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')