"""
Frame Preprocessing Module for ALPR System
Handles frame resizing, grayscale conversion, blur, and other preprocessing operations.
"""

import cv2
import numpy as np
from typing import Tuple


class FramePreprocessor:
    """Preprocesses video frames for license plate detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target size for frame resizing (width, height)
        """
        self.target_size = target_size
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to grayscale.
        
        Args:
            frame: Input frame
            
        Returns:
            Grayscale frame
        """
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def apply_blur(self, frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.
        
        Args:
            frame: Input frame
            kernel_size: Size of the Gaussian kernel (must be odd)
            
        Returns:
            Blurred frame
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def apply_contrast_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            frame: Input frame (grayscale or color)
            
        Returns:
            Contrast-enhanced frame
        """
        if len(frame.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back to BGR
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(frame)
    
    def preprocess_for_detection(self, frame: np.ndarray, apply_blur: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for detection.
        
        Args:
            frame: Input frame
            apply_blur: Whether to apply blur
            
        Returns:
            Preprocessed frame
        """
        # Resize
        frame = self.resize_frame(frame)
        
        # Enhance contrast
        frame = self.apply_contrast_enhancement(frame)
        
        # Apply blur if specified
        if apply_blur:
            frame = self.apply_blur(frame)
        
        return frame
    
    def preprocess_for_ocr(self, plate_roi: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate region for OCR.
        
        Args:
            plate_roi: Cropped plate region
            
        Returns:
            Preprocessed plate region
        """
        # Convert to grayscale
        gray = self.to_grayscale(plate_roi)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply binary threshold
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame values to 0-1 range.
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame
        """
        return frame.astype(np.float32) / 255.0
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detection dicts with keys: 'box', 'confidence', 'class', 'plate_text'
            
        Returns:
            Frame with drawn boxes
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            confidence = detection.get('confidence', 0)
            class_name = detection.get('class', 'Unknown')
            plate_text = detection.get('plate_text', '')
            
            # Draw rectangle
            color = (0, 255, 0) if class_name == 'license_plate' else (255, 0, 0)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            if plate_text:
                label += f" | {plate_text}"
            
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy
