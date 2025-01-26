from flask import Flask, render_template, Response
import cv2
import torch

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model_path = 'best.pt'  # Replace with your actual model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam


def generate_frames():
    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break

        # Run YOLOv5 model on the frame
        results = model(frame)  # Inference
        results.render()  # Render bounding boxes and labels onto the frame

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for live streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Render the index page
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Serve the live video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)