
import cv2
import numpy as np
from typing import List, Dict

try:
    from .enhanced_detection import enhanced_plate_detection
except ImportError:
    from enhanced_detection import enhanced_plate_detection


def enhance_detector_with_fallback(detector_instance, frame: np.ndarray) -> List[Dict]:
    """
    Enhance the existing detector with additional fallback strategies.
    
    Args:
        detector_instance: Instance of ALPRDetector
        frame: Input frame
        
    Returns:
        List of detected license plates in the detector's expected format
    """
    
    # First, try the existing detection pipeline
    detections = detector_instance.get_license_plate_detections(frame)
    
    # Extract plates only (it returns (vehicles, plates) tuple)
    if isinstance(detections, tuple):
        plates = detections[1]
    else:
        plates = detections
    
    # If no plates found, use enhanced detection methods
    if not plates or len(plates) == 0:
        print("[DEBUG] No plates found by standard detection, trying enhanced methods...")
        enhanced_plates = enhanced_plate_detection(frame, min_confidence=0.3)
        
        if enhanced_plates:
            print(f"[DEBUG] Enhanced detection found {len(enhanced_plates)} plate(s)")
            # Convert to detector's expected format
            plates = []
            for p in enhanced_plates:
                plate_dict = {
                    'box': p['box'],
                    'confidence': p['confidence'],
                    'class': 'license_plate',
                    'class_id': -1,  # Unknown class from enhanced detection
                    'class_name': 'license_plate',
                    'source': p['method']
                }
                plates.append(plate_dict)
    
    return plates


def analyze_frame_for_issues(frame: np.ndarray) -> Dict:
    """
    Analyze frame to identify why detection might be failing.
    
    Args:
        frame: Input frame to analyze
        
    Returns:
        Dictionary with analysis results
    """
    
    analysis = {
        'frame_size': frame.shape,
        'frame_channels': frame.shape[2] if len(frame.shape) > 2 else 1,
        'brightness_levels': {},
        'contrast': 0,
        'edge_density': 0,
        'potential_issues': []
    }
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    analysis['brightness_levels'] = {
        'mean': float(mean_brightness),
        'std': float(std_brightness),
        'min': float(np.min(gray)),
        'max': float(np.max(gray))
    }
    
    # Calculate contrast (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    contrast = np.var(laplacian)
    analysis['contrast'] = float(contrast)
    
    # Calculate edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    analysis['edge_density'] = float(edge_density)
    
    # Identify potential issues
    if mean_brightness < 50:
        analysis['potential_issues'].append("Frame too dark (mean brightness < 50)")
    elif mean_brightness > 200:
        analysis['potential_issues'].append("Frame too bright (mean brightness > 200)")
    
    if std_brightness < 20:
        analysis['potential_issues'].append("Low contrast (std < 20)")
    
    if contrast < 100:
        analysis['potential_issues'].append("Very low edge contrast")
    
    if edge_density < 0.01:
        analysis['potential_issues'].append("Very few edges detected")
    elif edge_density > 0.5:
        analysis['potential_issues'].append("Too many edges (possible noise)")
    
    return analysis


def preprocess_frame_for_detection(frame: np.ndarray) -> np.ndarray:
    """
    Apply comprehensive preprocessing to improve detection.
    
    Args:
        frame: Input frame
        
    Returns:
        Preprocessed frame
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to smooth while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Convert back to BGR for detector
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    # Example usage
    print("Detector Enhancement Module")
    print("Import this module to use enhanced detection capabilities")
