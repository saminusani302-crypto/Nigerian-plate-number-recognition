"""
Flask Web Application for ALPR System
Provides web interface for real-time license plate recognition.
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from pathlib import Path
import threading
import json
from datetime import datetime
import io
import base64

from alpr_system import ALPRPipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize ALPR pipeline
try:
    alpr = ALPRPipeline(log_dir='logs', confidence_threshold=0.5)
    app.logger.info("ALPR Pipeline initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize ALPR: {e}")
    alpr = None

# Global variables for streaming
streaming_active = False
stream_thread = None
latest_frame = None
streaming_results = {}


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    stats = alpr.get_statistics()
    
    return jsonify({
        'status': 'ok',
        'frames_processed': stats.get('frames_processed', 0),
        'logger_stats': stats.get('logger_stats', {}),
        'streaming_active': streaming_active
    })


@app.route('/api/process-image', methods=['POST'])
def process_image():
    """Process uploaded image."""
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        # Read image
        file_stream = io.BytesIO(file.read())
        file_array = np.frombuffer(file_stream.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        
        # Process
        results = alpr.process_frame(frame, log_results=True)
        
        # Prepare response
        response = {
            'status': 'ok',
            'frame_number': results['frame_number'],
            'timestamp': results['timestamp'],
            'detections': results['detections'],
            'plates': results['plates']
        }
        
        # Encode visualization
        if results['visualization_frame'] is not None:
            _, buffer = cv2.imencode('.jpg', results['visualization_frame'])
            response['visualization'] = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process uploaded video."""
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        # Save video temporarily
        temp_path = f'temp_video_{datetime.now().timestamp()}.mp4'
        file.save(temp_path)
        
        # Process video
        results = alpr.process_video(temp_path, max_frames=300, display=False)
        
        # Clean up
        Path(temp_path).unlink(missing_ok=True)
        
        return jsonify({
            'status': 'ok',
            'total_frames': results['total_frames'],
            'plates_detected': results['plates_detected'],
            'unique_plates': results['unique_plates'],
            'processing_time': results['processing_time'],
            'average_fps': results['average_fps'],
            'plates': results['plates']
        })
    
    except Exception as e:
        app.logger.error(f"Error processing video: {e}")
        Path(temp_path).unlink(missing_ok=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/start-webcam', methods=['POST'])
def start_webcam():
    """Start webcam streaming."""
    global streaming_active, stream_thread
    
    if streaming_active:
        return jsonify({'status': 'error', 'message': 'Streaming already active'}), 400
    
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    streaming_active = True
    stream_thread = threading.Thread(target=_webcam_stream_worker, daemon=True)
    stream_thread.start()
    
    return jsonify({'status': 'ok', 'message': 'Webcam streaming started'})


@app.route('/api/stop-webcam', methods=['POST'])
def stop_webcam():
    """Stop webcam streaming."""
    global streaming_active
    
    streaming_active = False
    
    return jsonify({'status': 'ok', 'message': 'Webcam streaming stopped'})


def _webcam_stream_worker():
    """Worker thread for webcam streaming."""
    global latest_frame, streaming_results
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        app.logger.error("Could not open webcam")
        return
    
    frame_count = 0
    
    try:
        while streaming_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame to maintain speed
            if frame_count % 5 == 0:
                results = alpr.process_frame(frame, log_results=True)
                
                # Store latest results
                streaming_results = {
                    'frame_number': results['frame_number'],
                    'timestamp': results['timestamp'],
                    'detections': results['detections'],
                    'plates': results['plates']
                }
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', results['visualization_frame'])
                latest_frame = base64.b64encode(buffer).decode('utf-8')
    
    finally:
        cap.release()


@app.route('/api/webcam-stream', methods=['GET'])
def webcam_stream():
    """Get latest webcam frame and results."""
    if latest_frame is None:
        return jsonify({'status': 'error', 'message': 'No frame available'}), 400
    
    response = {
        'status': 'ok',
        'frame': latest_frame,
        'results': streaming_results
    }
    
    return jsonify(response)


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get logged plates."""
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    format_type = request.args.get('format', 'all')
    logs = alpr.get_logs(format=format_type)
    
    return jsonify({
        'status': 'ok',
        'count': len(logs),
        'logs': logs
    })


@app.route('/api/logs/export', methods=['GET'])
def export_logs():
    """Export logs as CSV or JSON."""
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    export_format = request.args.get('format', 'csv')
    
    try:
        export_path = f'plates_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format}'
        alpr.logger.export_logs(export_path, format=export_format)
        
        return send_file(export_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """Clear all logs."""
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    try:
        success = alpr.logger.clear_logs()
        
        if success:
            return jsonify({'status': 'ok', 'message': 'Logs cleared'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to clear logs'}), 500
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get detailed statistics."""
    if alpr is None:
        return jsonify({'status': 'error', 'message': 'ALPR not initialized'}), 500
    
    stats = alpr.get_statistics()
    logger_stats = alpr.logger.get_statistics()
    
    return jsonify({
        'status': 'ok',
        'frames_processed': stats.get('frames_processed', 0),
        'logger_stats': logger_stats,
        'streaming_active': streaming_active
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'status': 'error', 'message': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
