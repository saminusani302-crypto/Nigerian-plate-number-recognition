"""
Nigerian ALPR System - Comprehensive Diagnostic Script
Identifies why plates are not being detected and provides fixes
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

print("\n" + "="*70)
print("🔍 NIGERIAN ALPR SYSTEM - COMPREHENSIVE DIAGNOSTIC")
print("="*70)

# Step 1: Check environment
print("\n1️⃣  CHECKING ENVIRONMENT...")
print("-" * 70)

try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  ✗ PyTorch not found")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("  ✓ YOLOv8 (ultralytics) imported")
except ImportError:
    print("  ✗ YOLOv8 not found")
    sys.exit(1)

try:
    import easyocr
    print("  ✓ EasyOCR imported")
except ImportError:
    print("  ✗ EasyOCR not found")
    sys.exit(1)

# Step 2: Check model files
print("\n2️⃣  CHECKING MODEL FILES...")
print("-" * 70)

model_path = Path('yolov8n.pt')
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ YOLOv8 model found: {size_mb:.1f} MB")
else:
    print(f"  ⚠️  YOLOv8 model not found at {model_path}")
    print("     Model will auto-download on first use (~33 MB)")

# Step 3: Test model loading
print("\n3️⃣  TESTING MODEL LOADING...")
print("-" * 70)

try:
    print("  Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"  ✓ Model loaded successfully on {device.upper()}")
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    sys.exit(1)

# Step 4: Create test image
print("\n4️⃣  CREATING TEST IMAGE...")
print("-" * 70)

# Create a test image with a rectangle that could be a plate
test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
# Draw a rectangle representing a vehicle
cv2.rectangle(test_image, (100, 150), (400, 350), (50, 100, 150), -1)
# Draw a rectangle representing a plate
cv2.rectangle(test_image, (150, 280), (350, 320), (200, 200, 200), -1)
# Add some text to simulate plate
cv2.putText(test_image, 'ABC 123 XY', (160, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

print("  ✓ Test image created (480x640)")
print("    Contains: Vehicle rectangle + License plate")

# Step 5: Test YOLO detection
print("\n5️⃣  TESTING YOLO DETECTION...")
print("-" * 70)

try:
    print("  Running inference...")
    results = model(test_image, conf=0.1, verbose=False)
    detections = len(results[0].boxes)
    print(f"  ✓ Inference completed")
    print(f"  ✓ Detections found: {detections}")
    
    if detections > 0:
        print("\n  Detection details:")
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_name = results[0].names[int(box.cls[0])]
            print(f"    {i+1}. Class: {cls_name}, Confidence: {conf:.1%}")
    else:
        print("  ⚠️  No detections on test image")
except Exception as e:
    print(f"  ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()

# Step 6: Test ALPRDetector
print("\n6️⃣  TESTING ALPRDETECTOR...")
print("-" * 70)

try:
    from alpr_system.detector import ALPRDetector
    print("  ✓ ALPRDetector imported")
    
    detector = ALPRDetector(model_path='yolov8n.pt')
    print("  ✓ ALPRDetector initialized")
    
    detections = detector.detect_objects(test_image, conf=0.1)
    print(f"  ✓ Detector.detect_objects() called")
    print(f"    Detections: {len(detections)}")
    
    if detections:
        for det in detections:
            print(f"    - {det['class_name']}: {det['confidence']:.1%}")
except Exception as e:
    print(f"  ✗ ALPRDetector failed: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Test ALPRPipeline
print("\n7️⃣  TESTING ALPRPIPELINE...")
print("-" * 70)

try:
    from alpr_system.main import ALPRPipeline
    print("  ✓ ALPRPipeline imported")
    
    pipeline = ALPRPipeline(confidence_threshold=0.3)
    print("  ✓ ALPRPipeline initialized")
    
    print("  Processing test image...")
    results = pipeline.process_frame(test_image, log_results=False)
    
    print(f"  ✓ Pipeline processed frame")
    print(f"    - Frame number: {results.get('frame_number')}")
    print(f"    - Detections: {results.get('detections')}")
    print(f"    - Plates found: {len(results.get('plates', []))}")
    print(f"    - Processing time: {results.get('processing_time'):.2f}ms")
    
    if results.get('plates'):
        print("\n  Plates detected:")
        for plate in results['plates']:
            print(f"    - Text: {plate.get('text', 'N/A')}")
            print(f"    - Confidence: {plate.get('detection_confidence', 0):.1%}")
except Exception as e:
    print(f"  ✗ ALPRPipeline failed: {e}")
    import traceback
    traceback.print_exc()

# Step 8: Test with real image if available
print("\n8️⃣  TESTING WITH REAL IMAGES...")
print("-" * 70)

# Look for any image files in common locations
image_locations = [
    Path('test_images'),
    Path('sample_images'),
    Path('examples'),
    Path('.'),
]

found_images = []
for loc in image_locations:
    if loc.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            found_images.extend(loc.glob(ext))

if found_images:
    print(f"  Found {len(found_images)} image(s)")
    
    for img_path in found_images[:3]:  # Test first 3
        print(f"\n  Testing: {img_path.name}")
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    ✗ Could not read image")
                continue
            
            h, w = img.shape[:2]
            print(f"    Resolution: {w}x{h}")
            
            # Test with pipeline
            results = pipeline.process_frame(img, log_results=False)
            plates = results.get('plates', [])
            print(f"    Plates detected: {len(plates)}")
            
            if plates:
                for p in plates:
                    print(f"      - {p.get('text', 'N/A')}")
            else:
                print(f"    ⚠️  No plates detected")
        except Exception as e:
            print(f"    ✗ Error: {e}")
else:
    print("  No real test images found")
    print("  (Place images in test_images/ or sample_images/ folders)")

# Step 9: Summary and recommendations
print("\n9️⃣  DIAGNOSIS SUMMARY...")
print("=" * 70)

print("\n✅ POSSIBLE ISSUES AND SOLUTIONS:\n")

issues_found = False

# Check if model can detect anything
print("Issue 1: No detections at all")
print("-" * 70)
print("""
SYMPTOMS: No vehicles or plates detected on ANY image
POSSIBLE CAUSES:
  a) Model expects different input format
  b) Confidence threshold too high
  c) Image quality/resolution too low
  d) Model not properly trained for Nigerian plates

SOLUTIONS:
  1. Lower confidence threshold (try 0.1-0.3 instead of 0.5)
  2. Check image resolution (should be 640x640 or similar)
  3. Ensure images contain actual vehicles with visible plates
  4. Verify model is trained for vehicle/plate detection
  5. Try preprocessing: resize to standard size, increase contrast

RECOMMENDED NEXT STEPS:
  Run: python alpr_system/detector.py <image_path>
  This will show detailed detection information
""")

print("\nIssue 2: Vehicles detected but no plates")
print("-" * 70)
print("""
SYMPTOMS: System finds vehicles but cannot locate plates
POSSIBLE CAUSES:
  a) Heuristic plate detection not working properly
  b) Plate angle/orientation makes detection difficult
  c) Plate resolution too low for OCR
  d) Nigerian plate format not recognized

SOLUTIONS:
  1. Ensure plates are clearly visible in images
  2. Try higher resolution test images
  3. Check heuristic detection settings in detector.py
  4. Verify plates match Nigerian format (ABC 123 XY)
  5. Check edge detection sensitivity settings

RECOMMENDED NEXT STEPS:
  Modify heuristic detection thresholds in detector.py
  Add preprocessing: histogram equalization, contrast enhancement
""")

print("\nIssue 3: Plates detected but text not recognized")
print("-" * 70)
print("""
SYMPTOMS: Bounding boxes appear but OCR returns garbage text
POSSIBLE CAUSES:
  a) Plate ROI too small or at wrong angle
  b) EasyOCR not optimized for number plates
  c) Preprocessing before OCR inadequate
  d) Text cleaning logic too aggressive

SOLUTIONS:
  1. Improve plate ROI extraction (expand margins)
  2. Add preprocessing: binarization, denoising
  3. Fine-tune OCR confidence threshold
  4. Check text cleaning/validation logic
  5. Verify Nigerian plate format patterns

RECOMMENDED NEXT STEPS:
  Check ocr.py preprocessing functions
  Add deskewing and perspective transform
  Test OCR directly on plate images
""")

# Step 10: Recommend specific fixes
print("\n🔧 RECOMMENDED IMMEDIATE FIXES:\n")
print("""
1. LOWER CONFIDENCE THRESHOLD
   Current: 0.5 (might be too high)
   Recommended: 0.3 for initial testing
   File: alpr_system/main.py line ~29
   Change: confidence_threshold=0.5 → 0.3

2. IMPROVE PREPROCESSING
   Add these to preprocessor.py:
   - Histogram equalization (CLAHE)
   - Morphological operations
   - Contrast enhancement
   - Image resizing to standard size

3. DEBUG OUTPUT
   Run with verbose logging:
   streamlit run alpr_system/ui/app.py --logger.level=debug

4. TEST DETECTION DIRECTLY
   python test_image.py <image_path> --verbose

5. VERIFY WITH SAVED DETECTIONS
   Check logs directory for saved detection results
""")

print("\n" + "="*70)
print("✓ DIAGNOSTIC COMPLETE")
print("="*70)
print("\nNext Step: Review issues above and apply recommended fixes")
print("Questions? Check IMPLEMENTATION_GUIDE.md for troubleshooting\n")
