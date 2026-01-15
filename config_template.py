"""
Configuration file for ALPR System
Copy this to config.py and modify settings as needed
"""

# Flask Configuration
FLASK_DEBUG = False
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_ENV = 'production'

# ALPR Configuration
ALPR_CONFIG = {
    'model_path': 'yolov8n.pt',  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
    'log_dir': 'logs',
    'confidence_threshold': 0.5,
    'device': 'cuda',  # Options: 'cuda', 'cpu'
}

# Detection Settings
DETECTOR_CONFIG = {
    'vehicle_classes': ['car', 'vehicle', 'truck', 'bus', 'motorcycle', 'bicycle'],
    'plate_classes': ['license_plate', 'plate', 'number_plate'],
    'min_confidence': 0.4,
    'iou_threshold': 0.5,
}

# OCR Settings
OCR_CONFIG = {
    'languages': ['en'],
    'gpu': True,
    'confidence_threshold': 0.3,
}

# Preprocessing Settings
PREPROCESSING_CONFIG = {
    'target_size': (640, 640),
    'blur_kernel_size': 5,
    'apply_contrast_enhancement': True,
    'clahe_clip_limit': 3.0,
}

# Logging Settings
LOGGING_CONFIG = {
    'format': 'csv',  # Options: 'csv', 'json', 'both'
    'include_visualization': True,
    'export_format': 'csv',
}

# Video Processing Settings
VIDEO_CONFIG = {
    'max_frames': None,  # None for all frames
    'skip_frames': 1,  # Process every Nth frame
    'display': True,
    'save_output': True,
}

# Webcam Settings
WEBCAM_CONFIG = {
    'camera_index': 0,
    'fps': 30,
    'resolution': (1280, 720),
    'skip_frames': 5,  # Process every Nth frame for speed
}

# Security Settings
SECURITY_CONFIG = {
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_image_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    'allowed_video_formats': ['mp4', 'avi', 'mov', 'mkv', 'flv'],
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'batch_size': 1,
    'num_workers': 4,
    'enable_benchmarking': False,
    'benchmark_iterations': 10,
}

# Nigerian Plate Format
NIGERIAN_PLATE_FORMAT = {
    'pattern': r'^[A-Z]{2,3}\s?\d{3}\s?[A-Z]{2}$',
    'example': 'ABC 123 XY',
    'description': '2-3 state letters, 3 digits, 2 type letters',
}

def get_config():
    """Get configuration dictionary"""
    return {
        'flask': FLASK_CONFIG,
        'alpr': ALPR_CONFIG,
        'detector': DETECTOR_CONFIG,
        'ocr': OCR_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'logging': LOGGING_CONFIG,
        'video': VIDEO_CONFIG,
        'webcam': WEBCAM_CONFIG,
        'security': SECURITY_CONFIG,
        'performance': PERFORMANCE_CONFIG,
    }
