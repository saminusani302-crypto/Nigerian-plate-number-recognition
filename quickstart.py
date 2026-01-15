#!/usr/bin/env python3
"""
Quick Start Script for Nigerian ALPR System
Run this to test the system with a sample image
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from alpr_system import ALPRPipeline
import cv2
import numpy as np


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║   Nigerian Automatic License Plate Recognition System      ║
    ║                      ALPR v1.0.0                           ║
    ║                                                            ║
    ║   Powered by YOLOv8 & EasyOCR                             ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def test_system():
    """Test ALPR system initialization."""
    print("🔧 Initializing ALPR System...")
    
    try:
        # Initialize pipeline
        alpr = ALPRPipeline(
            model_path='yolov8n.pt',
            log_dir='logs',
            confidence_threshold=0.5,
            device='cpu'  # Use 'cuda' if GPU available
        )
        
        print("✓ ALPR Pipeline initialized successfully")
        print(f"✓ Detection Device: {alpr.detector.device}")
        
        # Get model info
        model_info = alpr.detector.get_model_info()
        print(f"✓ Model: YOLOv8")
        print(f"✓ Input Size: {model_info['input_size']}")
        print(f"✓ Classes: {len(model_info['class_names'])} detected")
        
        # Test with dummy image
        print("\n📷 Testing with dummy image...")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[100:200, 150:350] = [255, 255, 255]  # Add white rectangle
        
        results = alpr.process_frame(dummy_image, log_results=False)
        print(f"✓ Frame processed in {results['processing_time']:.3f} seconds")
        print(f"✓ Detections found: {results['detections']}")
        
        # Get statistics
        stats = alpr.get_statistics()
        print(f"\n📊 System Statistics:")
        print(f"  • Frames processed: {stats['frames_processed']}")
        print(f"  • Logger stats: {stats['logger_stats']}")
        
        print("\n✓ System test completed successfully!")
        print("\n🚀 Next steps:")
        print("  1. Start Flask: python app.py")
        print("  2. Open browser: http://localhost:5000")
        print("  3. Upload images or start webcam streaming")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during initialization: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Check Python version: python3 --version (should be 3.9+)")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Check GPU availability (optional): nvidia-smi")
        return False


def example_usage():
    """Show example usage."""
    print("\n" + "="*60)
    print("📚 Example Usage")
    print("="*60)
    
    example_code = '''
from alpr_system import ALPRPipeline

# Initialize
alpr = ALPRPipeline(confidence_threshold=0.5)

# Process image
results = alpr.process_image('image.jpg')
for plate in results['plates']:
    print(f"Detected: {plate['formatted_text']}")

# Process video
stats = alpr.process_video('video.mp4', display=True)
print(f"Unique plates: {stats['unique_plates']}")

# Access logs
logs = alpr.get_logs(format='valid')
for log in logs:
    print(f"{log['timestamp']}: {log['formatted_plate']}")
    '''
    
    print(example_code)


if __name__ == '__main__':
    print_banner()
    
    if test_system():
        example_usage()
    else:
        sys.exit(1)
