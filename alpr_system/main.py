"""
Main ALPR Pipeline
Integrates all components: detection, preprocessing, OCR, and logging.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime

from .detector import ALPRDetector
from .ocr import PlateOCR
from .preprocessor import FramePreprocessor
from .logger import ALPRLogger


class ALPRPipeline:
    """Complete Automatic License Plate Recognition pipeline."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', log_dir: str = 'logs',
                 confidence_threshold: float = 0.5, device: Optional[str] = None):
        """
        Initialize the ALPR pipeline.
        
        Args:
            model_path: Path to YOLOv8 model
            log_dir: Directory for logs
            confidence_threshold: Detection confidence threshold
            device: Device for inference ('cpu' or 'cuda')
        """
        self.detector = ALPRDetector(model_path=model_path, device=device)
        self.detector.set_confidence_threshold(confidence_threshold)
        
        self.ocr = PlateOCR()
        self.preprocessor = FramePreprocessor()
        self.logger = ALPRLogger(log_dir=log_dir)
        
        self.frame_count = 0
        self.last_log_time = 0
        self.min_log_interval = 0.5  # Minimum seconds between logging same plate
        
        self.results_history = []
    
    def process_frame(self, frame: np.ndarray, log_results: bool = True) -> Dict:
        """
        Process a single frame through the ALPR pipeline.
        
        Args:
            frame: Input frame (BGR)
            log_results: Whether to log detected plates
            
        Returns:
            Dict with processing results
        """
        self.frame_count += 1
        start_time = time.time()
        
        # Initialize results dict
        results = {
            'frame_number': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'frame_size': frame.shape,
            'detections': [],
            'plates': [],
            'processing_time': 0.0,
            'visualization_frame': None
        }
        
        try:
            # Step 1: Preprocessing
            preprocessed = self.preprocessor.preprocess_for_detection(frame)
            
            # Step 2: Vehicle and Plate Detection
            vehicles, plates = self.detector.get_license_plate_detections(
                preprocessed, vehicle_conf=0.5
            )
            
            # Step 3: Also run detection on original frame for better results
            vehicles_orig, plates_orig = self.detector.get_license_plate_detections(
                frame, vehicle_conf=0.4
            )
            
            # Combine detections
            all_plates = plates + plates_orig
            all_plates = self._deduplicate_detections(all_plates)
            
            results['detections'] = {
                'vehicles': len(vehicles) + len(vehicles_orig),
                'plates': len(all_plates)
            }
            
            # Step 4: Extract and process plate regions
            plate_regions = self.detector.extract_plate_regions(frame, all_plates)
            
            for i, plate_info in enumerate(plate_regions):
                roi = plate_info['roi']
                box = plate_info['box']
                detection = plate_info['detection']
                
                # Preprocess plate region for OCR
                plate_processed = self.preprocessor.preprocess_for_ocr(roi)
                
                # Step 5: OCR
                plate_text, ocr_confidence = self.ocr.extract_plate_text(
                    plate_processed, confidence_threshold=0.3
                )
                
                # Validate
                is_valid = self.ocr.validate_nigerian_plate(plate_text)
                formatted = self.ocr.format_nigerian_plate(plate_text) if is_valid else plate_text
                
                plate_result = {
                    'index': i,
                    'box': box,
                    'raw_text': plate_text,
                    'formatted_text': formatted,
                    'ocr_confidence': ocr_confidence,
                    'detection_confidence': detection.get('confidence', 0.0),
                    'is_valid': is_valid
                }
                
                results['plates'].append(plate_result)
                
                # Step 6: Log results
                if log_results and is_valid:
                    self.logger.log_plate(
                        plate_number=plate_text,
                        formatted_plate=formatted,
                        confidence=ocr_confidence,
                        is_valid=is_valid,
                        frame_source='video',
                        frame_index=self.frame_count,
                        detection_confidence=detection.get('confidence', 0.0)
                    )
            
            # Step 7: Create visualization
            vis_frame = frame.copy()
            vis_frame = self._draw_results(vis_frame, all_plates, results['plates'])
            results['visualization_frame'] = vis_frame
            
            # Update history
            self.results_history.append(results)
            if len(self.results_history) > 100:
                self.results_history.pop(0)
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error processing frame: {e}")
        
        finally:
            results['processing_time'] = time.time() - start_time
        
        return results
    
    def _deduplicate_detections(self, detections: List[Dict], 
                               iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate detections using NMS.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Deduplicated detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for detection in detections:
            is_duplicate = False
            
            for kept in keep:
                iou = self._calculate_iou(detection['box'], kept['box'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(detection)
        
        return keep
    
    @staticmethod
    def _calculate_iou(box1: List, box2: List) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _draw_results(self, frame: np.ndarray, plate_detections: List[Dict],
                     plate_results: List[Dict]) -> np.ndarray:
        """
        Draw detection and OCR results on frame.
        
        Args:
            frame: Input frame
            plate_detections: List of plate detections
            plate_results: List of OCR results
            
        Returns:
            Frame with drawn results
        """
        frame_vis = frame.copy()
        
        # Draw plate detections
        for i, detection in enumerate(plate_detections):
            box = detection['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Find corresponding OCR result
            ocr_result = None
            for result in plate_results:
                if result['index'] == i:
                    ocr_result = result
                    break
            
            # Draw box
            color = (0, 255, 0) if (ocr_result and ocr_result['is_valid']) else (0, 165, 255)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 3)
            
            # Draw text
            if ocr_result:
                label = f"{ocr_result['formatted_text']} ({ocr_result['ocr_confidence']:.2f})"
                bg_color = (0, 150, 0) if ocr_result['is_valid'] else (0, 100, 150)
            else:
                label = f"Plate {detection['confidence']:.2f}"
                bg_color = (100, 100, 100)
            
            # Draw label with background
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x1
            text_y = max(20, y1 - 10)
            
            cv2.rectangle(frame_vis, (text_x, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5), bg_color, -1)
            cv2.putText(frame_vis, label, (text_x + 2, text_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Draw frame info
        info = f"Frame: {self.frame_count} | Plates: {len(plate_detections)} | Time: {time.time():.2f}"
        cv2.putText(frame_vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame_vis
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None,
                     display: bool = True, output_path: Optional[str] = None) -> Dict:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            display: Whether to display video
            output_path: Path to save output video
            
        Returns:
            Dict with processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        plates_detected = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if max_frames and frame_count > max_frames:
                    break
                
                # Process frame
                results = self.process_frame(frame, log_results=True)
                
                # Collect plates
                for plate in results['plates']:
                    if plate['is_valid']:
                        plates_detected.append(plate['formatted_text'])
                
                # Display
                if display:
                    vis_frame = results['visualization_frame']
                    cv2.imshow('ALPR', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save to output video
                if writer:
                    writer.write(results['visualization_frame'])
                
                # Print progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    print(f"Processed {frame_count}/{total_frames} frames at {fps_actual:.2f} FPS")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        
        return {
            'total_frames': frame_count,
            'plates_detected': len(plates_detected),
            'unique_plates': len(set(plates_detected)),
            'processing_time': elapsed_time,
            'average_fps': frame_count / elapsed_time if elapsed_time > 0 else 0,
            'plates': list(set(plates_detected))
        }
    
    def process_image(self, image_path: str, save_output: bool = True) -> Dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to image file
            save_output: Whether to save output image
            
        Returns:
            Dict with processing results
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        results = self.process_frame(frame, log_results=True)
        
        # Save output if requested
        if save_output and results['visualization_frame'] is not None:
            output_path = Path(image_path).stem + '_alpr.jpg'
            cv2.imwrite(output_path, results['visualization_frame'])
            results['output_image'] = str(output_path)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'frames_processed': self.frame_count,
            'logger_stats': self.logger.get_statistics(),
            'recent_results': self.results_history[-10:] if self.results_history else []
        }
    
    def get_logs(self, format: str = 'all') -> List[Dict]:
        """Get logged plates."""
        if format == 'valid':
            return self.logger.get_valid_plates()
        return self.logger.get_all_logs()
