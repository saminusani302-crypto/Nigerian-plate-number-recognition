
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch


class ALPRDetector:
    """Detects vehicles and license plates using YOLOv8."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: Optional[str] = None):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run inference on ('cpu', 'cuda', etc.). Auto-detect if None.
        """
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Confidence thresholds
        self.conf_threshold = 0.5
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set confidence threshold for detections.
        
        Args:
            threshold: Confidence threshold (0-1)
        """
        self.conf_threshold = max(0, min(1, threshold))
    
    def detect_objects(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in frame using YOLOv8.
        
        Args:
            frame: Input frame
            conf: Confidence threshold (uses self.conf_threshold if None)
            
        Returns:
            List of detections with keys: 'box', 'confidence', 'class', 'class_name'
        """
        if conf is None:
            conf = self.conf_threshold
        
        # Run inference
        results = self.model(frame, conf=conf, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf_score,
                    'class': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def filter_detections_by_class(self, detections: List[Dict], 
                                   class_names: List[str]) -> List[Dict]:
        """
        Filter detections by class name.
        
        Args:
            detections: List of detection dicts
            class_names: List of class names to keep
            
        Returns:
            Filtered detections
        """
        return [d for d in detections if d['class_name'] in class_names]
    
    def get_license_plate_detections(self, frame: np.ndarray, 
                                     vehicle_conf: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect vehicles and license plates from frame.
        Uses YOLOv8 detection and falls back to heuristic methods if no plates found.
        
        Args:
            frame: Input frame
            vehicle_conf: Confidence threshold for vehicles
            
        Returns:
            Tuple of (vehicle_detections, plate_detections)
        """
        all_detections = self.detect_objects(frame, conf=vehicle_conf)
        
        # Separate vehicles and license plates
        vehicles = self.filter_detections_by_class(all_detections, ['car', 'vehicle', 'truck', 'bus', 'motorcycle', 'bicycle'])
        plates = self.filter_detections_by_class(all_detections, ['license_plate', 'plate', 'number_plate'])
        
        # If no specific plate detections, try fallback methods
        if not plates and vehicles:
            # Use heuristic detection for plates within vehicle regions
            plates = self._detect_plates_heuristic(frame, vehicles)
        
        return vehicles, plates
    
    def _detect_plates_heuristic(self, frame: np.ndarray, 
                                 vehicle_detections: List[Dict]) -> List[Dict]:
        """
        Detect license plates using heuristic methods within vehicle regions.
        Uses edge detection and contour analysis to find rectangular plate regions.
        
        Args:
            frame: Input frame
            vehicle_detections: List of vehicle detections
            
        Returns:
            List of potential license plate detections
        """
        plates = []
        h, w = frame.shape[:2]
        
        for vehicle in vehicle_detections:
            box = vehicle['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Ensure box is within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extract vehicle region
            vehicle_roi = frame[y1:y2, x1:x2]
            
            if vehicle_roi.size == 0:
                continue
            
            # Look for rectangular regions that match license plate characteristics
            candidate_plates = self._find_plate_candidates(vehicle_roi)
            
            # Adjust coordinates to frame reference
            for candidate in candidate_plates:
                candidate['box'] = [
                    candidate['box'][0] + x1,
                    candidate['box'][1] + y1,
                    candidate['box'][2] + x1,
                    candidate['box'][3] + y1
                ]
                plates.append(candidate)
        
        return plates
    
    def _find_plate_candidates(self, roi: np.ndarray, 
                              min_area: float = 500, 
                              max_area: float = None) -> List[Dict]:
        """
        Find license plate candidates using edge detection and contour analysis.
        
        Args:
            roi: Region of interest (vehicle region)
            min_area: Minimum contour area
            max_area: Maximum contour area (calculated if None)
            
        Returns:
            List of candidate plate detections
        """
        candidates = []
        h, w = roi.shape[:2]
        
        if max_area is None:
            max_area = (h * w) * 0.5  # Max 50% of ROI area
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Check aspect ratio (license plates are wider than tall)
            aspect_ratio = cw / (ch + 1e-5)
            
            # Nigerian license plates are typically 3:1 or 4:1 aspect ratio
            if 2.0 <= aspect_ratio <= 5.0:
                candidates.append({
                    'box': [x, y, x + cw, y + ch],
                    'confidence': 0.6,  # Heuristic confidence
                    'class': -1,  # Special marker for heuristic detection
                    'class_name': 'license_plate_heuristic'
                })
        
        return candidates
    
    def extract_plate_regions(self, frame: np.ndarray, 
                            plate_detections: List[Dict]) -> List[Dict]:
        """
        Extract license plate regions from frame.
        
        Args:
            frame: Input frame
            plate_detections: List of plate detections
            
        Returns:
            List of dicts with keys: 'roi', 'box', 'detection'
        """
        h, w = frame.shape[:2]
        plate_regions = []
        
        for detection in plate_detections:
            box = detection['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Add padding around plate (10% of plate size)
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            roi = frame[y1:y2, x1:x2].copy()
            
            if roi.size > 0:
                plate_regions.append({
                    'roi': roi,
                    'box': [x1, y1, x2, y2],
                    'detection': detection
                })
        
        return plate_regions
    
    def detect_from_file(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (frame, detections)
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        detections = self.detect_objects(frame)
        return frame, detections
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model information
        """
        return {
            'model_type': self.model.task,
            'device': self.device,
            'class_names': self.model.names
        }
    
    def set_model_parameters(self, **kwargs):
        """
        Set model parameters for inference.
        
        Args:
            **kwargs: Parameters to set (e.g., imgsz, conf, iou)
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
    
    def benchmark(self, frame: np.ndarray, iterations: int = 10) -> Dict:
        """
        Benchmark detection speed.
        
        Args:
            frame: Input frame
            iterations: Number of iterations
            
        Returns:
            Dict with benchmark results
        """
        import time
        
        # Warmup
        self.detect_objects(frame)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            self.detect_objects(frame)
            times.append(time.time() - start)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }
