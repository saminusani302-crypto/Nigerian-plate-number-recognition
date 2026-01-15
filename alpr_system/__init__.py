"""
Nigerian License Plate Recognition System
A complete ALPR system using YOLOv8 and EasyOCR
"""

from .main import ALPRPipeline
from .detector import ALPRDetector
from .ocr import PlateOCR
from .preprocessor import FramePreprocessor
from .logger import ALPRLogger

__version__ = '1.0.0'
__all__ = [
    'ALPRPipeline',
    'ALPRDetector',
    'PlateOCR',
    'FramePreprocessor',
    'ALPRLogger'
]
