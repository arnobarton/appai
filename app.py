from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import os

# Initialize Flask app
app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello, Render!"
    
# Load YOLOv5 model
model_path = 'best.pt'  # Replace with your actual model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Directories
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Generate frames for live video
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)  # YOLOv5 inference
        results.render()  # Render bounding boxes and labels

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run YOLOv5 inference on the uploaded file
    results = model(file_path)
    results.save(save_dir=UPLOAD_FOLDER)

    # Return the processed file URL
    processed_file_url = f"/static/processed/{file.filename}"
    return jsonify({'status': 'success', 'processed_file_url': processed_file_url})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'online'})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
