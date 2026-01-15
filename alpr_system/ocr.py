"""
Optical Character Recognition Module for License Plates
Uses EasyOCR to extract text from Nigerian license plates.
"""

import cv2
import numpy as np
import easyocr
import re
from typing import List, Dict, Tuple, Optional
import torch


class PlateOCR:
    """Extracts text from license plates using EasyOCR."""
    
    # Nigerian license plate format patterns
    NIGERIAN_PLATE_PATTERNS = [
        r'^[A-Z]{2,3}\s?\d{3}\s?[A-Z]{2}$',  # e.g., ABC 123 XY or ABCDE 123 XY
        r'^[A-Z]{2,3}\-?\d{3}\-?[A-Z]{2}$',  # e.g., ABC-123-XY
    ]
    
    def __init__(self, languages: List[str] = ['en'], gpu: Optional[bool] = None):
        """
        Initialize the OCR engine.
        
        Args:
            languages: List of language codes (e.g., ['en', 'fr'])
            gpu: Use GPU if available. Auto-detect if None.
        """
        if gpu is None:
            gpu = torch.cuda.is_available()
        
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        self.gpu = gpu
        self.languages = languages
    
    def extract_text_from_image(self, image: np.ndarray, 
                              confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image (BGR or grayscale)
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            List of dicts with keys: 'text', 'confidence', 'box'
        """
        # Convert grayscale to BGR if needed (EasyOCR expects 3-channel)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Run OCR
        results = self.reader.readtext(image)
        
        extractions = []
        for (box, text, confidence) in results:
            if confidence >= confidence_threshold:
                extractions.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'box': box
                })
        
        return extractions
    
    def extract_plate_text(self, plate_roi: np.ndarray, 
                         confidence_threshold: float = 0.4) -> Tuple[str, float]:
        """
        Extract and clean license plate text.
        
        Args:
            plate_roi: Cropped license plate region
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Tuple of (plate_text, confidence)
        """
        # Extract all text
        extractions = self.extract_text_from_image(plate_roi, confidence_threshold)
        
        if not extractions:
            return '', 0.0
        
        # Concatenate all detected text
        full_text = ' '.join([e['text'] for e in extractions])
        avg_confidence = np.mean([e['confidence'] for e in extractions])
        
        # Clean and format
        cleaned_text = self.clean_plate_text(full_text)
        
        return cleaned_text, avg_confidence
    
    def clean_plate_text(self, text: str) -> str:
        """
        Clean OCR output to match Nigerian plate format.
        
        Args:
            text: Raw OCR output
            
        Returns:
            Cleaned plate text (uppercase, uppercase letters and digits only)
        """
        # Convert to uppercase
        text = text.upper().strip()
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Replace common OCR errors
        replacements = {
            'O': '0',  # Capital O might be zero
            'I': '1',  # Capital I might be one
            'S': '5',  # Capital S might be five
            'B': '8',  # Capital B might be eight
            'L': '1',  # Capital L might be one
        }
        
        # Smart replacement - only replace if character is in wrong position
        # For now, keep as is - let's be conservative
        
        # Keep only uppercase letters and digits
        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
        
        # Format: ABC 123 XY (3 letters, space, 3 digits, space, 2 letters)
        # Try to identify the pattern
        clean = re.sub(r'[^\w]', '', text)  # Remove all non-word chars
        
        if len(clean) >= 8:  # Minimum for a plate
            # Try to extract letters and numbers
            letters_nums = re.findall(r'[A-Z0-9]', text)
            if letters_nums:
                text = ''.join(letters_nums)
        
        return text.strip()
    
    def validate_nigerian_plate(self, text: str) -> bool:
        """
        Validate if text matches Nigerian plate format.
        
        Args:
            text: Plate text to validate
            
        Returns:
            True if text matches Nigerian plate format
        """
        # Remove spaces and dashes for validation
        normalized = re.sub(r'[\s\-]', '', text)
        
        # Check if it's 8 characters (2-3 letters, 3 digits, 2 letters)
        if len(normalized) != 8 and len(normalized) != 7:
            return False
        
        # Should have at least some digits
        if not any(c.isdigit() for c in normalized):
            return False
        
        # Should have at least some letters
        if not any(c.isalpha() for c in normalized):
            return False
        
        return True
    
    def format_nigerian_plate(self, text: str) -> str:
        """
        Format text to standard Nigerian plate format.
        
        Args:
            text: Raw plate text
            
        Returns:
            Formatted plate text (e.g., ABC 123 XY)
        """
        # Clean text first
        text = self.clean_plate_text(text)
        
        # Remove spaces and dashes
        clean = re.sub(r'[\s\-]', '', text)
        
        # Extract letters and numbers
        letters = re.findall(r'[A-Z]', clean)
        numbers = re.findall(r'[0-9]', clean)
        
        # Format: 2-3 letters, 3 numbers, 2 letters
        if len(letters) >= 4 and len(numbers) >= 3:
            formatted = ''.join(letters[:3]) + ' ' + ''.join(numbers[:3]) + ' ' + ''.join(letters[3:5])
            return formatted
        
        return text
    
    def process_batch(self, plate_regions: List[np.ndarray], 
                     confidence_threshold: float = 0.4) -> List[Dict]:
        """
        Process multiple plate regions.
        
        Args:
            plate_regions: List of cropped plate images
            confidence_threshold: Minimum confidence
            
        Returns:
            List of dicts with OCR results
        """
        results = []
        for i, roi in enumerate(plate_regions):
            text, confidence = self.extract_plate_text(roi, confidence_threshold)
            
            results.append({
                'plate_number': text,
                'confidence': confidence,
                'valid': self.validate_nigerian_plate(text),
                'formatted': self.format_nigerian_plate(text)
            })
        
        return results
    
    def extract_from_file(self, image_path: str) -> List[Dict]:
        """
        Extract text from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of extracted text dicts
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        return self.extract_text_from_image(image)
    
    def draw_ocr_results(self, image: np.ndarray, 
                        ocr_results: List[Dict]) -> np.ndarray:
        """
        Draw OCR results on image.
        
        Args:
            image: Input image
            ocr_results: OCR results from extract_text_from_image
            
        Returns:
            Image with drawn OCR results
        """
        image_copy = image.copy()
        
        for result in ocr_results:
            box = result['box']
            text = result['text']
            confidence = result['confidence']
            
            # Convert box coordinates to integers
            pts = np.array(box, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw box
            cv2.polylines(image_copy, [pts], True, (0, 255, 0), 2)
            
            # Draw text
            label = f"{text} ({confidence:.2f})"
            cv2.putText(image_copy, label, tuple(map(int, box[0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_copy
