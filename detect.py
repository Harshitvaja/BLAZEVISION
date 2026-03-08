from flask import Flask, render_template, request, make_response, send_file, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision
import json
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
model = YOLO('best.pt')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            img = cv2.imread(filename)
            results = model(img)
            annotated_img = results[0].plot()
            _, img_encoded = cv2.imencode('.jpg', annotated_img)
            response = make_response(img_encoded.tobytes())
            response.headers['Content-Type'] = 'image/jpeg'
            return response

    return render_template("index.html")

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['file']
    nparr = np.fromstring(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(frame)
    annotated_frame = results[0].plot()
    _, img_encoded = cv2.imencode('.jpg', annotated_frame)
    response = make_response(img_encoded.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response

@app.route('/process_video', methods=['POST'])
def process_video():
    file = request.files['file']
    video_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        frames.append(annotated_frame)

    cap.release()

    # Combine frames into a video
    output_path = os.path.join(UPLOAD_FOLDER, 'output_video.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (annotated_frame.shape[1], annotated_frame.shape[0]))

    for frame in frames:
        out.write(frame)

    out.release()

    return send_file(output_path, mimetype='video/mp4')

@app.route('/get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    """Get list of uploaded files with metadata"""
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                # Get file stats
                stat = os.stat(file_path)
                file_size = stat.st_size
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                
                # Determine file type
                ext = filename.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                    file_type = 'image'
                elif ext in ['mp4', 'avi', 'mov', 'mkv']:
                    file_type = 'video'
                else:
                    file_type = 'other'
                
                files.append({
                    'name': filename,
                    'size': file_size,
                    'type': file_type,
                    'modified': modified_time.isoformat(),
                    'url': f'/uploads/{filename}'
                })
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify(files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
