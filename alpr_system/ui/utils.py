"""
UI Utility Functions for Nigerian ALPR System
Helper functions for image processing, data formatting, and UI interactions.
"""

import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io


class DetectionFormatter:
    """Format and prepare detection results for UI display."""
    
    # Nigerian plate type patterns and rules
    PLATE_TYPES = {
        'personal': {
            'color': 'blue',
            'bgr': (255, 0, 0),
            'pattern': r'^[A-Z]{2,3}\s?\d{3}\s?[A-Z]{2}$'
        },
        'commercial': {
            'color': 'red',
            'bgr': (0, 0, 255),
            'pattern': r'^[A-Z]{2,3}\s?\d{3}\s?[A-Z]{2}$'
        },
        'government': {
            'color': 'green',
            'bgr': (0, 255, 0),
            'pattern': r'^[A-Z]{2,3}\s?\d{3}\s?[A-Z]{2}$'
        }
    }
    
    # Nigerian states abbreviations
    NIGERIAN_STATES = {
        'AB': 'Abia', 'AD': 'Adamawa', 'AK': 'Akwa Ibom', 'AN': 'Anambra',
        'BA': 'Bauchi', 'BY': 'Bayelsa', 'BE': 'Benue', 'BO': 'Borno',
        'CR': 'Cross River', 'DE': 'Delta', 'EB': 'Ebonyi', 'ED': 'Edo',
        'EK': 'Ekiti', 'EN': 'Enugu', 'FC': 'Federal Capital Territory',
        'GO': 'Gombe', 'JI': 'Jigawa', 'KD': 'Kaduna', 'KN': 'Kano',
        'KT': 'Katsina', 'KE': 'Kebbi', 'KO': 'Kogi', 'LA': 'Lagos',
        'NA': 'Nasarawa', 'NI': 'Niger', 'OG': 'Ogun', 'ON': 'Ondo',
        'OS': 'Osun', 'OY': 'Oyo', 'PL': 'Plateau', 'RI': 'Rivers',
        'SO': 'Sokoto', 'TA': 'Taraba', 'YO': 'Yobe', 'ZA': 'Zamfara'
    }
    
    @staticmethod
    def extract_plate_info(plate_text: str, detection_confidence: float,
                          ocr_confidence: float = 0.0) -> Dict:
        """
        Extract structured information from plate text.
        
        Args:
            plate_text: Recognized plate number text
            detection_confidence: YOLO detection confidence
            ocr_confidence: OCR text confidence
            
        Returns:
            Dictionary with structured plate information
        """
        plate_text = str(plate_text).upper().strip()
        
        info = {
            'plate_text': plate_text,
            'detection_confidence': detection_confidence,
            'ocr_confidence': ocr_confidence,
            'overall_confidence': (detection_confidence + ocr_confidence) / 2,
            'plate_type': DetectionFormatter._determine_plate_type(plate_text),
            'state_code': DetectionFormatter._extract_state_code(plate_text),
            'vehicle_category': 'Unknown',
            'owner_name': 'Not Available in Database',
            'registration_date': 'Not Available',
        }
        
        return info
    
    @staticmethod
    def _determine_plate_type(plate_text: str) -> str:
        """Determine plate type (Personal/Commercial/Government)."""
        # Simplified heuristic - would be improved with ML model
        if any(gov_code in plate_text for gov_code in ['FIRS', 'NPA', 'NG']):
            return 'Government'
        elif len(plate_text) > 10:
            return 'Commercial'
        else:
            return 'Personal'
    
    @staticmethod
    def _extract_state_code(plate_text: str) -> str:
        """Extract Nigerian state code from plate."""
        # Extract last 2 characters (state code)
        if len(plate_text) >= 2:
            state_code = plate_text[-2:].upper()
            return state_code
        return 'NG'
    
    @staticmethod
    def format_detection_result(detection: Dict, frame_number: int = 0,
                               processing_time_ms: float = 0.0) -> Dict:
        """
        Format complete detection result for UI display.
        
        Args:
            detection: Raw detection dict from ALPR
            frame_number: Current frame number
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Formatted detection dictionary
        """
        plate_info = DetectionFormatter.extract_plate_info(
            detection.get('text', 'UNKNOWN'),
            detection.get('detection_confidence', 0.0),
            detection.get('ocr_confidence', 0.0)
        )
        
        formatted = {
            **plate_info,
            'box': detection.get('box', []),
            'detection_time': datetime.now().isoformat(),
            'frame_number': frame_number,
            'processing_time': processing_time_ms,
            'plate_color': DetectionFormatter._get_plate_color_name(plate_info['plate_type']),
        }
        
        return formatted
    
    @staticmethod
    def _get_plate_color_name(plate_type: str) -> str:
        """Get plate color name from type."""
        if 'commercial' in plate_type.lower():
            return 'Red'
        elif 'government' in plate_type.lower():
            return 'Green'
        else:
            return 'Blue'


class ImageAnnotator:
    """Draw annotations on images for visualization."""
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict],
                       draw_text: bool = True, thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (BGR)
            detections: List of detection dictionaries
            draw_text: Whether to draw text labels
            thickness: Line thickness
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for detection in detections:
            box = detection.get('box', [])
            if not box or len(box) < 4:
                continue
            
            x1, y1, x2, y2 = map(int, box[:4])
            plate_text = detection.get('plate_text', 'UNKNOWN')
            plate_type = detection.get('plate_type', 'Personal')
            confidence = detection.get('overall_confidence', 0.0)
            
            # Get color for plate type
            color = ImageAnnotator._get_color_bgr(plate_type)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            if draw_text:
                # Draw text background
                label = f"{plate_text} ({confidence:.1%})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x, text_y = x1, max(y1 - 10, text_size[1])
                
                # Draw background rectangle for text
                cv2.rectangle(
                    annotated,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
        
        return annotated
    
    @staticmethod
    def _get_color_bgr(plate_type: str) -> Tuple[int, int, int]:
        """Get BGR color tuple for plate type."""
        plate_type_lower = str(plate_type).lower()
        
        if 'commercial' in plate_type_lower or 'red' in plate_type_lower:
            return (0, 0, 255)  # Red
        elif 'government' in plate_type_lower or 'green' in plate_type_lower:
            return (0, 255, 0)  # Green
        else:
            return (255, 0, 0)  # Blue
    
    @staticmethod
    def draw_plate_card(plate_text: str, plate_type: str, confidence: float,
                       size: Tuple[int, int] = (400, 150)) -> Image.Image:
        """
        Create a styled plate card image.
        
        Args:
            plate_text: Plate number
            plate_type: Type of plate
            confidence: Recognition confidence
            size: Card size (width, height)
            
        Returns:
            PIL Image of the card
        """
        card = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(card)
        
        # Get color for plate type
        if 'commercial' in plate_type.lower():
            bg_color = (255, 50, 50)  # Red
        elif 'government' in plate_type.lower():
            bg_color = (50, 200, 50)  # Green
        else:
            bg_color = (50, 100, 255)  # Blue
        
        # Draw colored background
        draw.rectangle([(10, 10), (size[0]-10, 70)], fill=bg_color)
        
        # Draw plate text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36)
            text_bbox = draw.textbbox((0, 0), plate_text, font=font)
            text_x = (size[0] - (text_bbox[2] - text_bbox[0])) // 2
            draw.text((text_x, 20), plate_text, fill='white', font=font)
        except:
            # Fallback if font not available
            draw.text((50, 25), plate_text, fill='white')
        
        # Draw info text
        try:
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font_small = ImageFont.load_default()
        
        info_text = f"Type: {plate_type} | Confidence: {confidence:.1%}"
        draw.text((10, 85), info_text, fill='black', font=font_small)
        
        return card


class DataProcessor:
    """Process and format data for tables and exports."""
    
    @staticmethod
    def prepare_history_dataframe(history: List[Dict]) -> pd.DataFrame:
        """
        Prepare detection history for display.
        
        Args:
            history: List of detection dictionaries
            
        Returns:
            Formatted pandas DataFrame
        """
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        
        # Ensure correct column order and types
        columns_order = [
            'plate_text', 'plate_type', 'plate_color', 'owner_name',
            'state_code', 'detection_time', 'overall_confidence',
            'detection_confidence', 'ocr_confidence'
        ]
        
        # Keep only existing columns
        existing_cols = [col for col in columns_order if col in df.columns]
        df = df[existing_cols]
        
        # Format columns
        if 'detection_time' in df.columns:
            df['detection_time'] = pd.to_datetime(df['detection_time'])
            df = df.sort_values('detection_time', ascending=False)
        
        if 'overall_confidence' in df.columns:
            df['overall_confidence'] = df['overall_confidence'].apply(lambda x: f"{x:.1%}")
        
        return df
    
    @staticmethod
    def export_to_csv(history: List[Dict], filepath: Optional[str] = None) -> str:
        """
        Export detection history to CSV.
        
        Args:
            history: List of detection dictionaries
            filepath: Optional filepath to save. If None, returns CSV string.
            
        Returns:
            CSV string or filepath
        """
        df = DataProcessor.prepare_history_dataframe(history)
        
        if filepath:
            df.to_csv(filepath, index=False)
            return filepath
        else:
            return df.to_csv(index=False)
    
    @staticmethod
    def export_to_json(history: List[Dict], filepath: Optional[str] = None) -> str:
        """
        Export detection history to JSON.
        
        Args:
            history: List of detection dictionaries
            filepath: Optional filepath to save. If None, returns JSON string.
            
        Returns:
            JSON string or filepath
        """
        json_str = json.dumps(history, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return filepath
        else:
            return json_str
    
    @staticmethod
    def get_statistics(history: List[Dict]) -> Dict:
        """
        Calculate statistics from detection history.
        
        Args:
            history: List of detection dictionaries
            
        Returns:
            Dictionary of statistics
        """
        if not history:
            return {
                'total_detections': 0,
                'personal_plates': 0,
                'commercial_plates': 0,
                'government_plates': 0,
                'avg_confidence': 0.0,
                'unique_plates': 0,
                'detection_time_range': None
            }
        
        df = pd.DataFrame(history)
        
        stats = {
            'total_detections': len(df),
            'personal_plates': len(df[df.get('plate_type', '').str.contains('Personal', case=False, na=False)]),
            'commercial_plates': len(df[df.get('plate_type', '').str.contains('Commercial', case=False, na=False)]),
            'government_plates': len(df[df.get('plate_type', '').str.contains('Government', case=False, na=False)]),
            'avg_confidence': float(df.get('overall_confidence', [0]).mean()),
            'unique_plates': len(df.get('plate_text', []).unique()),
        }
        
        if 'detection_time' in df.columns:
            df['detection_time'] = pd.to_datetime(df['detection_time'])
            time_range = (df['detection_time'].max() - df['detection_time'].min()).total_seconds()
            stats['detection_time_range'] = f"{time_range:.1f}s"
        
        return stats


class PerformanceMonitor:
    """Monitor and log system performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.frame_times = []
        self.detection_times = []
        self.ocr_times = []
    
    def log_frame_time(self, elapsed_ms: float):
        """Log frame processing time."""
        self.frame_times.append({
            'timestamp': datetime.now(),
            'time_ms': elapsed_ms
        })
    
    def get_average_frame_time(self, window: int = 100) -> float:
        """Get average frame processing time."""
        if not self.frame_times:
            return 0.0
        return np.mean([t['time_ms'] for t in self.frame_times[-window:]])
    
    def get_fps(self, window: int = 100) -> float:
        """Calculate frames per second."""
        avg_time = self.get_average_frame_time(window)
        return 1000.0 / avg_time if avg_time > 0 else 0.0
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        return {
            'avg_frame_time_ms': self.get_average_frame_time(),
            'current_fps': self.get_fps(),
            'total_frames': len(self.frame_times),
            'min_frame_time_ms': min((t['time_ms'] for t in self.frame_times), default=0),
            'max_frame_time_ms': max((t['time_ms'] for t in self.frame_times), default=0),
        }


__all__ = [
    'DetectionFormatter',
    'ImageAnnotator',
    'DataProcessor',
    'PerformanceMonitor',
]
