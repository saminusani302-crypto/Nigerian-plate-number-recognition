"""
Example usage of the ALPR system
Demonstrates various ways to use the Nigerian License Plate Recognition system
"""

from alpr_system import ALPRPipeline
import cv2
from pathlib import Path


def example_1_process_single_image():
    """Example 1: Process a single image"""
    print("\n" + "="*50)
    print("Example 1: Process Single Image")
    print("="*50)
    
    # Initialize ALPR pipeline
    alpr = ALPRPipeline(log_dir='logs', confidence_threshold=0.5)
    
    # Process image
    image_path = 'sample_image.jpg'
    if Path(image_path).exists():
        results = alpr.process_image(image_path, save_output=True)
        
        print(f"Frame number: {results['frame_number']}")
        print(f"Detections: {results['detections']}")
        print(f"Plates found: {len(results['plates'])}")
        
        for i, plate in enumerate(results['plates']):
            print(f"  Plate {i+1}: {plate['formatted_text']} (confidence: {plate['ocr_confidence']:.2f})")
    else:
        print(f"Image not found: {image_path}")


def example_2_process_video():
    """Example 2: Process a video file"""
    print("\n" + "="*50)
    print("Example 2: Process Video File")
    print("="*50)
    
    alpr = ALPRPipeline(log_dir='logs', confidence_threshold=0.5)
    
    video_path = 'sample_video.mp4'
    if Path(video_path).exists():
        # Process video with display
        stats = alpr.process_video(
            video_path, 
            max_frames=100,  # Process first 100 frames
            display=True,
            output_path='output_video.mp4'
        )
        
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Plates detected: {stats['plates_detected']}")
        print(f"Unique plates: {stats['unique_plates']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Detected plates: {stats['plates']}")
    else:
        print(f"Video not found: {video_path}")


def example_3_webcam_streaming():
    """Example 3: Real-time webcam processing"""
    print("\n" + "="*50)
    print("Example 3: Webcam Real-time Processing")
    print("="*50)
    
    alpr = ALPRPipeline(log_dir='logs', confidence_threshold=0.5)
    
    cap = cv2.VideoCapture(0)  # Webcam
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    frame_count = 0
    print("Processing webcam feed (press 'q' to quit)...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for speed
            if frame_count % 5 == 0:
                results = alpr.process_frame(frame, log_results=True)
                
                # Display results
                vis_frame = results['visualization_frame']
                cv2.imshow('ALPR System', vis_frame)
                
                if results['plates']:
                    print(f"\nFrame {frame_count}: {len(results['plates'])} plate(s) detected")
                    for plate in results['plates']:
                        if plate['is_valid']:
                            print(f"  ✓ {plate['formatted_text']} (confidence: {plate['ocr_confidence']:.2f})")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def example_4_batch_processing():
    """Example 4: Batch process multiple images"""
    print("\n" + "="*50)
    print("Example 4: Batch Process Images")
    print("="*50)
    
    alpr = ALPRPipeline(log_dir='logs', confidence_threshold=0.5)
    
    image_dir = Path('images')
    if image_dir.exists():
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        print(f"Found {len(image_files)} images")
        
        all_plates = []
        for image_path in image_files:
            results = alpr.process_image(str(image_path), save_output=True)
            
            for plate in results['plates']:
                if plate['is_valid']:
                    all_plates.append(plate['formatted_text'])
                    print(f"{image_path.name}: {plate['formatted_text']}")
        
        print(f"\nTotal valid plates found: {len(all_plates)}")
        print(f"Unique plates: {len(set(all_plates))}")
    else:
        print(f"Images directory not found: {image_dir}")


def example_5_access_logs():
    """Example 5: Access and manage logs"""
    print("\n" + "="*50)
    print("Example 5: Access and Manage Logs")
    print("="*50)
    
    alpr = ALPRPipeline(log_dir='logs')
    
    # Get all logs
    all_logs = alpr.get_logs(format='all')
    print(f"Total log entries: {len(all_logs)}")
    
    # Get valid plates only
    valid_logs = alpr.get_logs(format='valid')
    print(f"Valid plates: {len(valid_logs)}")
    
    # Get statistics
    stats = alpr.logger.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Valid plates: {stats['valid_plates']}")
    print(f"  Unique plates: {stats.get('unique_plates', 0)}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    print(f"  Average detection confidence: {stats['average_detection_confidence']:.2f}")
    
    # Display recent logs
    if valid_logs:
        print("\nRecent valid plates:")
        for log in valid_logs[-5:]:
            print(f"  {log['timestamp']}: {log['formatted_plate']} ({log['confidence']:.2f})")
    
    # Export logs
    alpr.logger.export_logs('exported_logs.csv', format='csv')
    print("\nLogs exported to: exported_logs.csv")


def example_6_custom_configuration():
    """Example 6: Custom configuration"""
    print("\n" + "="*50)
    print("Example 6: Custom Configuration")
    print("="*50)
    
    # Initialize with custom settings
    alpr = ALPRPipeline(
        model_path='yolov8m.pt',  # Larger model for better accuracy
        log_dir='custom_logs',
        confidence_threshold=0.6,  # Higher threshold for more confident detections
        device='cuda'  # Use GPU
    )
    
    print("ALPR Pipeline initialized with custom configuration:")
    print(f"  Model: yolov8m.pt")
    print(f"  Log directory: custom_logs")
    print(f"  Confidence threshold: 0.6")
    print(f"  Device: cuda")
    
    # Get model info
    model_info = alpr.detector.get_model_info()
    print(f"\nModel Information:")
    print(f"  Type: {model_info['model_type']}")
    print(f"  Device: {model_info['device']}")
    print(f"  Input size: {model_info['input_size']}")


def example_7_search_logs():
    """Example 7: Search and filter logs"""
    print("\n" + "="*50)
    print("Example 7: Search and Filter Logs")
    print("="*50)
    
    alpr = ALPRPipeline(log_dir='logs')
    
    # Search by date
    date_str = '2024-01-15'
    logs_by_date = alpr.logger.get_logs_by_date(date_str)
    print(f"Logs from {date_str}: {len(logs_by_date)}")
    
    # Search by plate
    plate_number = 'ABC123XY'
    logs_by_plate = alpr.logger.get_logs_by_plate(plate_number)
    print(f"Logs containing plate {plate_number}: {len(logs_by_plate)}")
    
    if logs_by_plate:
        for log in logs_by_plate:
            print(f"  {log['timestamp']}: {log['formatted_plate']}")


def main():
    """Run examples"""
    print("Nigerian ALPR System - Usage Examples")
    print("====================================\n")
    
    examples = [
        ("1", "Process single image", example_1_process_single_image),
        ("2", "Process video file", example_2_process_video),
        ("3", "Webcam streaming", example_3_webcam_streaming),
        ("4", "Batch process images", example_4_batch_processing),
        ("5", "Access logs", example_5_access_logs),
        ("6", "Custom configuration", example_6_custom_configuration),
        ("7", "Search logs", example_7_search_logs),
    ]
    
    print("Available examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    
    print("\nTo run all examples, or select specific ones:")
    choice = input("\nEnter example number(s) (comma-separated, or 'all'): ").strip()
    
    if choice.lower() == 'all':
        for num, desc, func in examples:
            try:
                func()
            except Exception as e:
                print(f"Error in example {num}: {e}")
    else:
        for num_str in choice.split(','):
            num_str = num_str.strip()
            for num, desc, func in examples:
                if num == num_str:
                    try:
                        func()
                    except Exception as e:
                        print(f"Error in example {num}: {e}")
                    break


if __name__ == '__main__':
    main()
