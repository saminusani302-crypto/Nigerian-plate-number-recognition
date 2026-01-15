"""
Logging Module for License Plate Recognition
Saves recognized plate numbers with timestamps to CSV and optionally database.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import threading
import os


class ALPRLogger:
    """Logs recognized license plates with timestamps."""
    
    def __init__(self, log_dir: str = 'logs', csv_filename: str = 'plates_log.csv'):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            csv_filename: Name of the CSV log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / csv_filename
        self.json_path = self.log_dir / 'plates_log.json'
        
        self.lock = threading.Lock()
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.get_csv_headers())
                writer.writeheader()
    
    @staticmethod
    def get_csv_headers() -> List[str]:
        """Get CSV column headers."""
        return [
            'id',
            'timestamp',
            'plate_number',
            'formatted_plate',
            'confidence',
            'is_valid',
            'frame_source',
            'frame_index',
            'detection_confidence',
            'notes'
        ]
    
    def log_plate(self, plate_number: str, formatted_plate: str = '', 
                 confidence: float = 0.0, is_valid: bool = False,
                 frame_source: str = 'webcam', frame_index: int = 0,
                 detection_confidence: float = 0.0, notes: str = '') -> Dict:
        """
        Log a recognized plate.
        
        Args:
            plate_number: Raw plate number from OCR
            formatted_plate: Formatted plate number
            confidence: OCR confidence score
            is_valid: Whether plate is valid
            frame_source: Source of frame (webcam, image, video, etc.)
            frame_index: Frame index in video
            detection_confidence: YOLOv8 detection confidence
            notes: Additional notes
            
        Returns:
            Dict with log entry
        """
        timestamp = datetime.now()
        entry_id = self._generate_id()
        
        log_entry = {
            'id': entry_id,
            'timestamp': timestamp.isoformat(),
            'plate_number': plate_number,
            'formatted_plate': formatted_plate,
            'confidence': round(confidence, 4),
            'is_valid': is_valid,
            'frame_source': frame_source,
            'frame_index': frame_index,
            'detection_confidence': round(detection_confidence, 4),
            'notes': notes
        }
        
        # Thread-safe logging
        with self.lock:
            self._write_to_csv(log_entry)
            self._write_to_json(log_entry)
        
        return log_entry
    
    def _generate_id(self) -> str:
        """Generate unique ID for log entry."""
        return datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    def _write_to_csv(self, entry: Dict):
        """Write log entry to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.get_csv_headers())
            writer.writerow(entry)
    
    def _write_to_json(self, entry: Dict):
        """Write log entry to JSON file."""
        entries = []
        
        # Read existing entries
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                entries = []
        
        # Add new entry
        entries.append(entry)
        
        # Write back
        with open(self.json_path, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def get_all_logs(self) -> List[Dict]:
        """
        Get all log entries.
        
        Returns:
            List of log entries
        """
        entries = []
        
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return entries
    
    def get_logs_by_date(self, date_str: str) -> List[Dict]:
        """
        Get log entries for a specific date.
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            List of log entries for that date
        """
        all_logs = self.get_all_logs()
        
        return [
            entry for entry in all_logs
            if entry['timestamp'].startswith(date_str)
        ]
    
    def get_logs_by_plate(self, plate_number: str) -> List[Dict]:
        """
        Get all logs for a specific plate.
        
        Args:
            plate_number: Plate number to search for
            
        Returns:
            List of log entries for that plate
        """
        all_logs = self.get_all_logs()
        
        return [
            entry for entry in all_logs
            if plate_number.upper() in entry['plate_number'].upper() or 
               plate_number.upper() in entry['formatted_plate'].upper()
        ]
    
    def get_valid_plates(self) -> List[Dict]:
        """
        Get all valid plates that were recognized.
        
        Returns:
            List of valid plate entries
        """
        all_logs = self.get_all_logs()
        return [entry for entry in all_logs if entry['is_valid']]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics from logs.
        
        Returns:
            Dict with various statistics
        """
        all_logs = self.get_all_logs()
        
        if not all_logs:
            return {
                'total_detections': 0,
                'valid_plates': 0,
                'average_confidence': 0.0,
                'average_detection_confidence': 0.0
            }
        
        valid_plates = [e for e in all_logs if e['is_valid']]
        
        return {
            'total_detections': len(all_logs),
            'valid_plates': len(valid_plates),
            'average_confidence': round(
                sum(e['confidence'] for e in all_logs) / len(all_logs), 4
            ),
            'average_detection_confidence': round(
                sum(e['detection_confidence'] for e in all_logs) / len(all_logs), 4
            ),
            'unique_plates': len(set(e['formatted_plate'] for e in valid_plates if e['formatted_plate']))
        }
    
    def export_logs(self, export_path: str, format: str = 'csv') -> bool:
        """
        Export logs to file.
        
        Args:
            export_path: Path to export file
            format: Export format ('csv' or 'json')
            
        Returns:
            True if export successful
        """
        try:
            all_logs = self.get_all_logs()
            
            if format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(all_logs, f, indent=2)
            else:  # csv
                with open(export_path, 'w', newline='') as f:
                    if all_logs:
                        writer = csv.DictWriter(f, fieldnames=self.get_csv_headers())
                        writer.writeheader()
                        writer.writerows(all_logs)
            
            return True
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return False
    
    def clear_logs(self) -> bool:
        """
        Clear all logs.
        
        Returns:
            True if successful
        """
        try:
            # Clear CSV
            self._initialize_csv()
            
            # Clear JSON
            with open(self.json_path, 'w') as f:
                json.dump([], f)
            
            return True
        except Exception as e:
            print(f"Error clearing logs: {e}")
            return False
    
    def get_log_file_path(self, format: str = 'csv') -> str:
        """
        Get path to log file.
        
        Args:
            format: File format ('csv' or 'json')
            
        Returns:
            Path to log file
        """
        if format.lower() == 'json':
            return str(self.json_path)
        return str(self.csv_path)
    
    def log_batch(self, plates: List[Dict]) -> List[Dict]:
        """
        Log multiple plates at once.
        
        Args:
            plates: List of plate dicts with keys: 'plate_number', 'confidence', etc.
            
        Returns:
            List of log entries
        """
        entries = []
        for plate in plates:
            entry = self.log_plate(
                plate_number=plate.get('plate_number', ''),
                formatted_plate=plate.get('formatted_plate', ''),
                confidence=plate.get('confidence', 0.0),
                is_valid=plate.get('is_valid', False),
                frame_source=plate.get('frame_source', 'batch'),
                frame_index=plate.get('frame_index', 0),
                detection_confidence=plate.get('detection_confidence', 0.0),
                notes=plate.get('notes', '')
            )
            entries.append(entry)
        
        return entries
