"""
Enhanced Detection Fix for Nigerian ALPR System
Provides improved plate detection using multiple strategies
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple

def enhanced_plate_detection(frame: np.ndarray, 
                           min_confidence: float = 0.3) -> List[Dict]:
    """
    Enhanced plate detection using multiple strategies.
    
    Args:
        frame: Input image
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detected plates with bounding boxes
    """
    plates = []
    h, w = frame.shape[:2]
    
    # Strategy 1: Color-based detection (Nigerian plates are typically white/yellow on blue/red)
    plates.extend(_detect_by_color(frame))
    
    # Strategy 2: Edge-based detection with multiple thresholds
    plates.extend(_detect_by_edges(frame))
    
    # Strategy 3: Contour analysis with aspect ratio filtering
    plates.extend(_detect_by_contours(frame))
    
    # Remove duplicates and low-confidence detections
    plates = _merge_and_filter_plates(plates, min_confidence)
    
    return plates


def _detect_by_color(frame: np.ndarray) -> List[Dict]:
    """
    Detect plates by color characteristics.
    Nigerian plates have white/yellow background with blue/red sides.
    """
    plates = []
    h, w = frame.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    
    # Look for high-saturation, high-value regions (bright colors)
    # This catches both blue-backed and red-backed plates
    mask = cv2.inRange(v_chan, 150, 255)  # High value (brightness)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area (typical plate is 5000-50000 pixels at 640x640)
        if area < 2000 or area > 80000:
            continue
        
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect = cw / (ch + 1e-5)
        
        # Nigerian plates: width/height typically 3-5
        if 2.5 <= aspect <= 5.5:
            plates.append({
                'box': [x, y, x + cw, y + ch],
                'confidence': 0.55,
                'method': 'color'
            })
    
    return plates


def _detect_by_edges(frame: np.ndarray) -> List[Dict]:
    """
    Detect plates using multiple edge detection strategies.
    """
    plates = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Strategy 1: Canny with different thresholds
    for canny_low, canny_high in [(50, 150), (100, 200), (150, 250)]:
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Enhance edges with morphological ops
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Dilate to connect nearby edges
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1500 or area > 100000:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / (ch + 1e-5)
            
            # Check for plate-like proportions
            if 2.0 <= aspect <= 6.0:
                # Calculate confidence based on contour quality
                rect = cv2.minAreaRect(contour)
                box_area = rect[1][0] * rect[1][1]
                fill_ratio = area / (box_area + 1e-5)
                
                confidence = 0.50 + (fill_ratio * 0.2)  # 0.5-0.7
                
                plates.append({
                    'box': [x, y, x + cw, y + ch],
                    'confidence': min(confidence, 0.7),
                    'method': f'edges_{canny_low}_{canny_high}'
                })
    
    return plates


def _detect_by_contours(frame: np.ndarray) -> List[Dict]:
    """
    Detect plates using contour analysis on multi-scale processing.
    """
    plates = []
    h, w = frame.shape[:2]
    
    # Try multiple preprocessing strategies
    for preprocessing in [_prep_histogram_eq, _prep_bilateral, _prep_threshold]:
        processed = preprocessing(frame)
        
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000 or area > 90000:
                continue
            
            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / (ch + 1e-5)
            
            # Plate-like dimensions
            if 2.2 <= aspect <= 5.8:
                # Approximate polygon
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Rectangles should have ~4 vertices
                vertices = len(approx)
                
                confidence = 0.50
                if vertices <= 6:  # Close to 4 (allows some distortion)
                    confidence = 0.65
                
                plates.append({
                    'box': [x, y, x + cw, y + ch],
                    'confidence': confidence,
                    'method': f'contour_v{vertices}'
                })
    
    return plates


def _prep_histogram_eq(frame: np.ndarray) -> np.ndarray:
    """Histogram equalization preprocessing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Edge detection
    edges = cv2.Canny(enhanced, 100, 200)
    return edges


def _prep_bilateral(frame: np.ndarray) -> np.ndarray:
    """Bilateral filter preprocessing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Bilateral filter preserves edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    # Edge detection
    edges = cv2.Canny(filtered, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def _prep_threshold(frame: np.ndarray) -> np.ndarray:
    """Threshold-based preprocessing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh


def _merge_and_filter_plates(plates: List[Dict], 
                            min_confidence: float = 0.3) -> List[Dict]:
    """
    Merge overlapping detections and filter by confidence.
    """
    if not plates:
        return []
    
    # Filter by confidence
    plates = [p for p in plates if p['confidence'] >= min_confidence]
    
    # Remove duplicates using NMS-like approach
    merged = []
    for plate in sorted(plates, key=lambda x: x['confidence'], reverse=True):
        is_duplicate = False
        
        for existing in merged:
            # Check if boxes overlap significantly
            if _boxes_overlap(plate['box'], existing['box'], threshold=0.3):
                is_duplicate = True
                break
        
        if not is_duplicate:
            merged.append(plate)
    
    return merged


def _boxes_overlap(box1: List[float], box2: List[float], 
                  threshold: float = 0.3) -> bool:
    """
    Check if two boxes overlap significantly.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return False  # No intersection
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou >= threshold


if __name__ == "__main__":
    # Test the enhanced detection
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        frame = cv2.imread(img_path)
        if frame is not None:
            print(f"Testing enhanced detection on {img_path}")
            plates = enhanced_plate_detection(frame, min_confidence=0.3)
            print(f"Found {len(plates)} plates")
            for i, plate in enumerate(plates):
                print(f"  {i+1}. Box: {[int(x) for x in plate['box']]}, "
                      f"Confidence: {plate['confidence']:.2f}, Method: {plate['method']}")
        else:
            print(f"Could not read image: {img_path}")
    else:
        print("Usage: python enhanced_detection.py <image_path>")
