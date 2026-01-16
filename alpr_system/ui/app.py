"""
Nigerian Automatic License Plate Recognition (ALPR) System
Professional Streamlit Dashboard - Main Application
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import traceback
from typing import Dict, List, Optional
import json
import time

# Import ALPR system
from alpr_system.main import ALPRPipeline
from alpr_system.ui.components import (
    apply_custom_theme,
    render_header,
    create_input_section,
    render_detection_display,
    create_vehicle_info_panel,
    create_logs_table,
    render_error_feedback,
    create_statistics_dashboard,
    create_sidebar_controls,
    COLORS,
)


# ============================================================================
# SESSION STATE AND INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'alpr' not in st.session_state:
        try:
            st.session_state.alpr = ALPRPipeline(
                log_dir='logs',
                confidence_threshold=0.3
            )
            st.session_state.alpr_initialized = True
        except Exception as e:
            st.session_state.alpr_initialized = False
            st.session_state.alpr_error = str(e)
    
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    if 'current_detections' not in st.session_state:
        st.session_state.current_detections = []
    
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    
    if 'camera_enabled' not in st.session_state:
        st.session_state.camera_enabled = False


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def process_image_file(uploaded_file, confidence_threshold: float = 0.5) -> Dict:
    """Process uploaded image file through ALPR pipeline."""
    try:
        # Read image
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'error': 'Invalid image file'}
        
        # Update confidence threshold
        st.session_state.alpr.detector.set_confidence_threshold(confidence_threshold)
        
        # Process frame
        results = st.session_state.alpr.process_frame(image, log_results=True)
        
        return {
            'success': True,
            'image': image,
            'results': results,
            'detections': results.get('plates', []),
            'processing_time': results.get('processing_time', 0)
        }
    
    except Exception as e:
        return {
            'error': f'Image processing failed: {str(e)}',
            'traceback': traceback.format_exc()
        }


def process_video_file(uploaded_file, confidence_threshold: float = 0.5, 
                       max_frames: int = 100) -> Dict:
    """Process uploaded video file through ALPR pipeline."""
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {'error': 'Failed to open video file'}
        
        # Update confidence threshold
        st.session_state.alpr.detector.set_confidence_threshold(confidence_threshold)
        
        # Process frames
        all_detections = []
        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for faster processing
            frame = cv2.resize(frame, (960, 540))
            
            # Process frame
            results = st.session_state.alpr.process_frame(frame, log_results=True)
            all_detections.extend(results.get('plates', []))
            
            frame_count += 1
            progress = min(frame_count / max_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_count} / {max_frames} frames")
        
        cap.release()
        Path(tmp_path).unlink()
        
        return {
            'success': True,
            'frames_processed': frame_count,
            'detections': all_detections,
            'total_unique_detections': len(set(d.get('plate_text') for d in all_detections))
        }
    
    except Exception as e:
        return {
            'error': f'Video processing failed: {str(e)}',
            'traceback': traceback.format_exc()
        }


def extract_detection_info(plate_info: Dict, detection_result: Dict = None) -> Dict:
    """Extract and format detection information."""
    detection_info = {
        'plate_text': plate_info.get('text', 'UNKNOWN'),
        'plate_confidence': plate_info.get('ocr_confidence', 0.0),
        'box': plate_info.get('box', []),
        'confidence': plate_info.get('detection_confidence', 0.0),
        'plate_type': plate_info.get('plate_type', 'Unknown'),
        'plate_color': plate_info.get('plate_color', 'Unknown'),
        'vehicle_category': plate_info.get('vehicle_category', 'Unknown'),
        'state_code': plate_info.get('state_code', 'NG'),
        'owner_name': plate_info.get('owner_name', 'Not Available'),
        'registration_date': plate_info.get('registration_date', 'Not Available'),
        'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'processing_time': plate_info.get('processing_time', 0),
        'frame_number': plate_info.get('frame_number', 0),
    }
    return detection_info


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    # Configure theme and page
    apply_custom_theme()
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Check ALPR initialization
    if not st.session_state.alpr_initialized:
        st.error(f"❌ Failed to initialize ALPR system: {st.session_state.alpr_error}")
        st.stop()
    
    # Create sidebar controls
    sidebar_settings = create_sidebar_controls()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        # Input section
        input_data = create_input_section()
        
        # Process based on input type
        if input_data['type'] == 'image' and input_data['file'] is not None:
            st.markdown("---")
            
            # Show processing status
            with st.spinner("🔄 Processing image..."):
                result = process_image_file(input_data['file'], input_data['confidence'])
            
            if 'error' in result:
                render_error_feedback('processing_error', result['error'])
            else:
                # Store data
                st.session_state.processed_image = result['image']
                st.session_state.current_detections = result['detections']
                
                # Display processing status
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Processing Time", f"{result['processing_time']:.0f}ms")
                with col2:
                    st.metric("🎯 Plates Detected", len(result['detections']))
                with col3:
                    st.metric("✅ Detection Rate", f"{min(100, len(result['detections']) * 50)}%")
                
                st.markdown("---")
                
                # Render detection display
                if result['detections']:
                    st.success(f"✅ Successfully detected {len(result['detections'])} license plate(s)!")
                    
                    render_detection_display(
                        result['image'],
                        result['detections'],
                        "📸 Image Detection Results"
                    )
                    
                    # Show detailed plate information
                    st.markdown("### 📋 Detected Plates")
                    for idx, plate in enumerate(result['detections'], 1):
                        with st.expander(f"🔷 Plate #{idx}: {plate.get('raw_text', 'Unknown')}", expanded=idx==1):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📝 Raw Text", plate.get('raw_text', 'N/A'))
                            with col2:
                                st.metric("✨ Formatted", plate.get('formatted_text', 'N/A'))
                            with col3:
                                st.metric("🔍 Confidence", f"{plate.get('ocr_confidence', 0):.0%}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                valid = "✅ Valid" if plate.get('is_valid') else "❌ Invalid"
                                st.metric("Status", valid)
                            with col2:
                                st.metric("Detection", f"{plate.get('detection_confidence', 0):.2f}")
                            with col3:
                                st.metric("Position", f"Box: {[int(x) for x in plate.get('box', [0,0,0,0])][:2]}")
                            
                            detection_info = extract_detection_info(plate)
                            st.json({
                                "Text": plate.get('raw_text'),
                                "Confidence": f"{plate.get('ocr_confidence', 0):.2%}",
                                "Valid": plate.get('is_valid'),
                                "Detected": datetime.now().isoformat()
                            })
                            
                            # Add to history
                            st.session_state.detection_history.append(detection_info)
                else:
                    st.warning("⚠️ No license plates detected in the image. Try:")
                    st.info("• Ensuring the plate is clearly visible\n• Better lighting conditions\n• Higher image quality\n• Different angle")
        
        elif input_data['type'] == 'video' and input_data['file'] is not None:
            st.markdown("---")
            
            with st.spinner("🔄 Processing video..."):
                result = process_video_file(input_data['file'], input_data['confidence'], max_frames=100)
            
            if 'error' in result:
                render_error_feedback('processing_error', result['error'])
            else:
                st.success(
                    f"✓ Video processed successfully\n\n"
                    f"Frames: {result['frames_processed']} | "
                    f"Detections: {len(result['detections'])} | "
                    f"Unique Plates: {result['total_unique_detections']}"
                )
                
                # Show summary
                if result['detections']:
                    st.session_state.detection_history.extend([
                        extract_detection_info(d) for d in result['detections']
                    ])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Frames Processed", result['frames_processed'])
                    with col2:
                        st.metric("Detections Found", len(result['detections']))
                    with col3:
                        st.metric("Unique Plates", result['total_unique_detections'])
                    
                    # Show recent detections
                    st.markdown("### 📋 Recent Detections from Video")
                    recent_df = pd.DataFrame([
                        extract_detection_info(d) for d in result['detections'][-10:]
                    ])
                    if not recent_df.empty:
                        st.dataframe(recent_df, use_container_width=True)
        
        elif input_data['type'] == 'camera' and input_data['enabled']:
            st.markdown("---")
            st.info("📹 Live camera feature coming soon! For now, use image or video upload.")
    
    with tab2:
        # Analytics dashboard
        create_statistics_dashboard(st.session_state.detection_history)
    
    with tab3:
        # Settings and system info
        st.markdown("## 🔧 Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Configuration")
            st.text_input("Model Path", value="yolov8n.pt", disabled=True)
            st.text_input("Device", value=sidebar_settings['device'], disabled=True)
            st.selectbox("Model Size", ["Nano", "Small", "Medium", "Large"])
        
        with col2:
            st.markdown("### Pipeline Settings")
            st.slider("Min Confidence", 0.0, 1.0, 0.3)
            st.slider("NMS Threshold", 0.0, 1.0, 0.45)
            st.slider("Max Detections per Image", 1, 100, 50)
        
        st.markdown("---")
        st.markdown("### System Information")
        
        if st.session_state.alpr_initialized:
            stats = st.session_state.alpr.get_statistics()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Frames Processed", stats.get('frames_processed', 0))
            with col2:
                st.metric("Total Detections", stats.get('plates_detected', 0))
            with col3:
                st.metric("Avg Processing Time", f"{stats.get('avg_processing_time', 0):.2f}ms")
    
    # Logs section at the bottom
    st.markdown("---")
    st.markdown("## 📋 Detection History")
    create_logs_table(st.session_state.detection_history)


if __name__ == "__main__":
    main()
