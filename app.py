from flask import Flask, render_template, request, jsonify, send_file, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import Counter
import uuid
import threading
import time
import json
import base64
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load YOLO model
try:
    model = YOLO("best.pt")
    labels = model.names
    logger.info(f"YOLO model loaded successfully with {len(labels)} classes")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None
    labels = {}

# Global variables for camera streaming and detection storage
camera = None
camera_active = False
current_detections = {}
detection_lock = threading.Lock()
stored_detections = {}  # Store detection data for highlighting

class CameraStream:
    def __init__(self):
        self.capture = None
        self.active = False
        self.colors = [
            (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184),
            (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)
        ]
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_detection_time = 0
        self.detection_interval = 0.5  # Run detection every 0.5 seconds
    
    def start(self, camera_index=0):
        """Start camera capture"""
        try:
            if self.capture is None:
                self.capture = cv2.VideoCapture(camera_index)
                if not self.capture.isOpened():
                    # Try different camera indices
                    for i in range(3):
                        self.capture = cv2.VideoCapture(i)
                        if self.capture.isOpened():
                            logger.info(f"Camera opened at index {i}")
                            break
                    else:
                        logger.error("No camera found")
                        return False
                
                # Set camera properties
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
                
            self.active = True
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture"""
        try:
            self.active = False
            if self.capture:
                self.capture.release()
                self.capture = None
            logger.info("Camera stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def get_frame(self):
        """Get current frame with detections"""
        if not self.active or self.capture is None:
            return None
        
        try:
            ret, frame = self.capture.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None
            
            # Store current frame for photo capture
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Run YOLO detection periodically to avoid performance issues
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_interval:
                self.last_detection_time = current_time
                self._run_detection(frame)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def _run_detection(self, frame):
        """Run YOLO detection on frame"""
        if model is None:
            return
        
        try:
            results = model(frame, verbose=False)[0].boxes
            detections = []
            counter = Counter()
            
            for i, det in enumerate(results):
                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                conf = det.conf.item()
                classidx = int(det.cls.item())
                
                if conf > 0.5:
                    classname = labels[classidx]
                    color = self.colors[classidx % len(self.colors)]
                    counter[classname] += 1
                    
                    detections.append({
                        'id': f'det_{i}',
                        'class': classname,
                        'conf': round(conf, 2),
                        'bbox': xyxy.tolist(),
                        'color': color
                    })
            
            # Update global detection data
            global current_detections
            with detection_lock:
                current_detections = {
                    'summary': dict(counter),
                    'detections': detections,
                    'timestamp': current_time
                }
                
        except Exception as e:
            logger.error(f"Error running detection: {e}")
    
    def capture_photo(self):
        """Capture current frame for photo analysis"""
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
        except Exception as e:
            logger.error(f"Error capturing photo: {e}")
        return None

def generate_frames():
    """Generate frames for camera streaming"""
    global camera, camera_active
    
    try:
        camera = CameraStream()
        
        if not camera.start():
            logger.error("Failed to start camera")
            return
        
        camera_active = True
        logger.info("Camera streaming started")
        
        while camera_active:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        logger.error(f"Error in frame generation: {e}")
    finally:
        if camera:
            camera.stop()
        camera_active = False
        logger.info("Camera streaming stopped")

def detect_image(image_path):
    """Detect objects in uploaded image"""
    if model is None:
        raise Exception("YOLO model not loaded")
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            raise Exception("Failed to load image")
        
        original_frame = frame.copy()
        results = model(frame, verbose=False)[0].boxes

        detections = []
        counter = Counter()
        colors = [
            (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184),
            (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)
        ]

        for i, det in enumerate(results):
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            conf = det.conf.item()
            classidx = int(det.cls.item())
            
            if conf > 0.5:
                classname = labels[classidx]
                color = colors[classidx % len(colors)]
                counter[classname] += 1

                # Draw bounding box and label
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), color, 2)
                label = f"{classname}: {int(conf * 100)}%"
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append({
                    'id': f'det_{i}',
                    'class': classname,
                    'conf': round(conf, 2),
                    'bbox': xyxy.tolist(),
                    'color': color
                })

        # Generate unique filename for the result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, frame)

        # Store original frame and detections for highlighting
        base_filename = result_filename.replace('.jpg', '')
        detection_data = {
            'original_frame': original_frame,
            'detections': detections,
            'colors': colors,
            'timestamp': time.time()
        }
        
        stored_detections[base_filename] = detection_data
        
        # Clean up old detection data (keep last 50)
        if len(stored_detections) > 50:
            oldest_key = min(stored_detections.keys(), 
                           key=lambda x: stored_detections[x]['timestamp'])
            del stored_detections[oldest_key]

        logger.info(f"Detected {len(detections)} objects in image")
        return result_filename, dict(counter), detections
        
    except Exception as e:
        logger.error(f"Error detecting image: {e}")
        raise

@app.route("/")
def index():
    """Serve main page"""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and detection"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files["image"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"upload_{timestamp}_{uuid.uuid4().hex[:8]}_{file.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
        file.save(image_path)
        
        logger.info(f"Image uploaded: {upload_filename}")
        
        # Process the image
        result_filename, counts, detections = detect_image(image_path)
        
        # Clean up upload file
        try:
            os.remove(image_path)
        except:
            pass
        
        return jsonify({
            "image_url": result_filename,
            "summary": counts,
            "detections": detections,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/capture_photo", methods=["POST"])
def capture_photo():
    """Capture photo from live camera feed and analyze it"""
    global camera
    
    try:
        if not camera or not camera_active:
            return jsonify({"error": "Camera is not active"}), 400
        
        # Capture current frame
        captured_frame = camera.capture_photo()
        if captured_frame is None:
            return jsonify({"error": "Failed to capture frame"}), 500
        
        # Save captured frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_filename = f"capture_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        capture_path = os.path.join(app.config['UPLOAD_FOLDER'], capture_filename)
        cv2.imwrite(capture_path, captured_frame)
        
        logger.info(f"Photo captured: {capture_filename}")
        
        # Analyze the captured image
        result_filename, counts, detections = detect_image(capture_path)
        
        # Clean up capture file
        try:
            os.remove(capture_path)
        except:
            pass
        
        return jsonify({
            "image_url": result_filename,
            "summary": counts,
            "detections": detections,
            "capture_source": "camera",
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/highlight/<base_filename>/<detection_id>")
def highlight_detection(base_filename, detection_id):
    """Generate highlighted version of image with specific detection emphasized"""
    try:
        if base_filename not in stored_detections:
            return jsonify({"error": "Detection data not found"}), 404
        
        data = stored_detections[base_filename]
        frame = data['original_frame'].copy()
        detections = data['detections']
        
        # Find the specific detection to highlight
        target_detection = None
        for det in detections:
            if det['id'] == detection_id:
                target_detection = det
                break
        
        if not target_detection:
            return jsonify({"error": "Detection not found"}), 404
        
        # Draw all detections normally first
        for det in detections:
            xyxy = det['bbox']
            color = det['color']
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), color, 2)
            label = f"{det['class']}: {int(det['conf'] * 100)}%"
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Highlight the target detection with glow effect
        xyxy = target_detection['bbox']
        glow_color = (0, 255, 255)  # Cyan glow
        
        # Create glow effect
        for thickness in range(12, 2, -2):
            alpha = 0.2 + (12 - thickness) * 0.05
            overlay = frame.copy()
            cv2.rectangle(overlay, tuple(xyxy[:2]), tuple(xyxy[2:]), glow_color, thickness)
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # Draw main highlight border
        cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), glow_color, 4)
        
        # Add highlight label
        highlight_label = f">>> {target_detection['class']}: {int(target_detection['conf'] * 100)}% <<<"
        label_size = cv2.getTextSize(highlight_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (xyxy[0], xyxy[1] - 40), (xyxy[0] + label_size[0], xyxy[1] - 10), 
                     glow_color, -1)
        cv2.putText(frame, highlight_label, (xyxy[0], xyxy[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Encode image to memory
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ret:
            return jsonify({"error": "Failed to encode image"}), 500
        
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
        
    except Exception as e:
        logger.error(f"Highlight error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/image/<filename>")
def serve_image(filename):
    """Serve processed images"""
    try:
        file_path = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Serve image error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera streaming"""
    global camera_active
    
    try:
        if not camera_active:
            # Start camera in a separate thread
            camera_thread = threading.Thread(target=lambda: list(generate_frames()), daemon=True)
            camera_thread.start()
            
            # Wait a moment for camera to initialize
            time.sleep(1)
            
            if camera_active:
                return jsonify({"status": "success", "message": "Camera started"})
            else:
                return jsonify({"status": "error", "message": "Failed to start camera"}), 500
        else:
            return jsonify({"status": "info", "message": "Camera already active"})
            
    except Exception as e:
        logger.error(f"Start camera error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera streaming"""
    global camera_active, camera
    
    try:
        camera_active = False
        if camera:
            camera.stop()
        
        return jsonify({"status": "success", "message": "Camera stopped"})
        
    except Exception as e:
        logger.error(f"Stop camera error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/camera_feed')
def camera_feed():
    """Stream camera feed"""
    try:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Camera feed error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera_detections')
def camera_detections():
    """Get current camera detections"""
    global current_detections
    try:
        with detection_lock:
            return jsonify(current_detections)
    except Exception as e:
        logger.error(f"Camera detections error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera_status')
def camera_status():
    """Get camera status"""
    return jsonify({
        "active": camera_active,
        "model_loaded": model is not None,
        "timestamp": time.time()
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

def cleanup_old_files():
    """Clean up old files in upload folder"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            current_time = time.time()
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    # Delete files older than 1 hour
                    if file_age > 3600:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=lambda: [
        time.sleep(300), cleanup_old_files()  # Clean every 5 minutes
    ], daemon=True)
    cleanup_thread.start()
    
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)