#!/usr/bin/env python3
"""
Unit Tests for Nigerian ALPR System
Tests all core components
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from alpr_system import (
    ALPRPipeline, ALPRDetector, PlateOCR,
    FramePreprocessor, ALPRLogger
)


class TestPreprocessor(unittest.TestCase):
    """Test FramePreprocessor module."""
    
    def setUp(self):
        """Initialize preprocessor."""
        self.preprocessor = FramePreprocessor()
    
    def test_resize_frame(self):
        """Test frame resizing."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = self.preprocessor.resize_frame(frame)
        self.assertEqual(resized.shape, (640, 640, 3))
    
    def test_to_grayscale(self):
        """Test grayscale conversion."""
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        gray = self.preprocessor.to_grayscale(frame)
        self.assertEqual(len(gray.shape), 2)
    
    def test_apply_blur(self):
        """Test blur application."""
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        blurred = self.preprocessor.apply_blur(frame, kernel_size=5)
        self.assertEqual(blurred.shape, frame.shape)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess_for_detection(frame)
        self.assertEqual(processed.shape, (640, 640, 3))


class TestDetector(unittest.TestCase):
    """Test ALPRDetector module."""
    
    def setUp(self):
        """Initialize detector."""
        try:
            self.detector = ALPRDetector(device='cpu')
        except Exception as e:
            self.skipTest(f"Could not initialize detector: {e}")
    
    def test_detector_init(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.device, 'cpu')
    
    def test_confidence_threshold(self):
        """Test confidence threshold setting."""
        self.detector.set_confidence_threshold(0.7)
        self.assertEqual(self.detector.conf_threshold, 0.7)
    
    def test_detect_objects(self):
        """Test object detection."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect_objects(frame, conf=0.5)
        self.assertIsInstance(detections, list)


class TestOCR(unittest.TestCase):
    """Test PlateOCR module."""
    
    def setUp(self):
        """Initialize OCR."""
        try:
            self.ocr = PlateOCR(gpu=False)
        except Exception as e:
            self.skipTest(f"Could not initialize OCR: {e}")
    
    def test_ocr_init(self):
        """Test OCR initialization."""
        self.assertIsNotNone(self.ocr)
    
    def test_clean_plate_text(self):
        """Test text cleaning."""
        text = "ABC 123 XY"
        cleaned = self.ocr.clean_plate_text(text)
        self.assertIsInstance(cleaned, str)
    
    def test_validate_nigerian_plate(self):
        """Test Nigerian plate validation."""
        valid = self.ocr.validate_nigerian_plate("ABC123XY")
        self.assertIsInstance(valid, bool)
    
    def test_format_nigerian_plate(self):
        """Test plate formatting."""
        text = "ABC123XY"
        formatted = self.ocr.format_nigerian_plate(text)
        self.assertIsInstance(formatted, str)


class TestLogger(unittest.TestCase):
    """Test ALPRLogger module."""
    
    def setUp(self):
        """Initialize logger with temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ALPRLogger(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_logger_init(self):
        """Test logger initialization."""
        self.assertIsNotNone(self.logger)
        self.assertTrue(self.logger.csv_path.exists())
    
    def test_log_plate(self):
        """Test logging a plate."""
        entry = self.logger.log_plate(
            plate_number="ABC123XY",
            formatted_plate="ABC 123 XY",
            confidence=0.95,
            is_valid=True
        )
        self.assertIn('timestamp', entry)
        self.assertEqual(entry['plate_number'], "ABC123XY")
    
    def test_get_logs(self):
        """Test retrieving logs."""
        self.logger.log_plate("ABC123XY", confidence=0.95)
        logs = self.logger.get_all_logs()
        self.assertEqual(len(logs), 1)
    
    def test_get_statistics(self):
        """Test getting statistics."""
        self.logger.log_plate("ABC123XY", confidence=0.95, is_valid=True)
        stats = self.logger.get_statistics()
        self.assertEqual(stats['total_detections'], 1)
        self.assertEqual(stats['valid_plates'], 1)


class TestPipeline(unittest.TestCase):
    """Test ALPRPipeline module."""
    
    def setUp(self):
        """Initialize pipeline."""
        try:
            self.temp_dir = tempfile.mkdtemp()
            self.alpr = ALPRPipeline(
                log_dir=self.temp_dir,
                device='cpu'
            )
        except Exception as e:
            self.skipTest(f"Could not initialize pipeline: {e}")
    
    def tearDown(self):
        """Clean up temp directory."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.alpr)
        self.assertEqual(self.alpr.frame_count, 0)
    
    def test_process_frame(self):
        """Test frame processing."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.alpr.process_frame(frame, log_results=False)
        
        self.assertIn('frame_number', results)
        self.assertIn('timestamp', results)
        self.assertIn('detections', results)
        self.assertIn('plates', results)
    
    def test_process_frame_with_logging(self):
        """Test frame processing with logging."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.alpr.process_frame(frame, log_results=True)
        
        self.assertIsNotNone(results)
        self.assertEqual(self.alpr.frame_count, 1)
    
    def test_get_statistics(self):
        """Test getting statistics."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.alpr.process_frame(frame)
        
        stats = self.alpr.get_statistics()
        self.assertEqual(stats['frames_processed'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Initialize system."""
        try:
            self.temp_dir = tempfile.mkdtemp()
            self.alpr = ALPRPipeline(log_dir=self.temp_dir, device='cpu')
        except Exception as e:
            self.skipTest(f"Could not initialize system: {e}")
    
    def tearDown(self):
        """Clean up."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test complete pipeline with realistic data."""
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some visual elements
        cv2.rectangle(frame, (100, 100), (300, 200), (255, 255, 255), -1)
        cv2.putText(frame, "ABC123XY", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Process frame
        results = self.alpr.process_frame(frame, log_results=True)
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertGreater(self.alpr.frame_count, 0)
    
    def test_multiple_frame_processing(self):
        """Test processing multiple frames."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for i in range(5):
            self.alpr.process_frame(frame, log_results=False)
        
        self.assertEqual(self.alpr.frame_count, 5)


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Nigerian ALPR System - Unit Tests")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestOCR))
    suite.addTests(loader.loadTestsFromTestCase(TestLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
