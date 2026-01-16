"""
Streamlit UI Components for Nigerian ALPR System
Professional dashboard with modern design and smooth interactions.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import tempfile
import json
import base64
from io import BytesIO

# Custom color scheme inspired by Nigerian traffic/security systems
COLORS = {
    'primary': '#1e3a8a',      # Deep blue
    'secondary': '#dc2626',    # Red (Nigerian police/security)
    'success': '#10b981',      # Green (Government plates)
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'light': '#f3f4f6',        # Light gray
    'dark': '#1f2937',         # Dark gray
    'personal_blue': '#3b82f6',      # Personal plate color
    'commercial_red': '#ef4444',     # Commercial plate color
    'government_green': '#10b981',   # Government plate color
}

def apply_custom_theme():
    """Apply custom Streamlit theme matching Nigerian color palette."""
    st.set_page_config(
        page_title="Nigerian ALPR System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Nigerian Automatic License Plate Recognition System v2.0",
        }
    )
    
    # Custom CSS for enhanced styling
    custom_css = f"""
    <style>
        :root {{
            --primary-color: {COLORS['primary']};
            --secondary-color: {COLORS['secondary']};
            --success-color: {COLORS['success']};
            --warning-color: {COLORS['warning']};
            --danger-color: {COLORS['danger']};
        }}
        
        * {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        /* Main container styling */
        .main {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
        }}
        
        /* Header styling */
        .header-title {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {COLORS['primary']};
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 0.5rem;
        }}
        
        .header-subtitle {{
            font-size: 1.1rem;
            color: #6b7280;
            font-weight: 500;
            margin-bottom: 2rem;
        }}
        
        /* Card styling */
        .metric-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 4px solid {COLORS['primary']};
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        
        /* Status indicators */
        .status-active {{
            color: {COLORS['success']};
            font-weight: 600;
            display: flex;
            align-items: center;
        }}
        
        .status-inactive {{
            color: #9ca3af;
            font-weight: 600;
            display: flex;
            align-items: center;
        }}
        
        /* Plate number styling */
        .plate-number {{
            font-family: 'Courier New', monospace;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 2px;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 4px;
            text-align: center;
        }}
        
        .plate-blue {{
            background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        }}
        
        .plate-red {{
            background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%);
        }}
        
        .plate-green {{
            background: linear-gradient(135deg, #10b981 0%, #065f46 100%);
        }}
        
        /* Detection box styling */
        .detection-box {{
            border: 3px solid {COLORS['success']};
            border-radius: 8px;
            padding: 1rem;
            background: rgba(16, 185, 129, 0.05);
            margin: 1rem 0;
        }}
        
        /* Table styling */
        .dataframe {{
            border-collapse: collapse;
            width: 100%;
        }}
        
        .dataframe th {{
            background: {COLORS['primary']};
            color: white;
            font-weight: 600;
            padding: 1rem;
            text-align: left;
        }}
        
        .dataframe td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .dataframe tbody tr:hover {{
            background: rgba(30, 58, 138, 0.05);
        }}
        
        /* Button styling */
        .stButton > button {{
            background: {COLORS['primary']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        }}
        
        .stButton > button:hover {{
            background: #1e40af;
            box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4);
            transform: translateY(-2px);
        }}
        
        /* Alert styling */
        .alert-success {{
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid {COLORS['success']};
            padding: 1rem;
            border-radius: 4px;
        }}
        
        .alert-warning {{
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid {COLORS['warning']};
            padding: 1rem;
            border-radius: 4px;
        }}
        
        .alert-error {{
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid {COLORS['danger']};
            padding: 1rem;
            border-radius: 4px;
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            padding: 1.5rem;
        }}
        
        /* Smooth animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .animate-fade-in {{
            animation: fadeIn 0.5s ease-in-out;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header():
    """Render professional header with status indicators."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(
            '<h1 class="header-title">🚗 Nigerian ALPR System</h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-subtitle">AI-Powered Vehicle Identification and Monitoring</p>',
            unsafe_allow_html=True
        )
    
    with col2:
        # Status indicators
        st.markdown("### System Status")
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            st.markdown(
                '<div class="status-active">✓ Camera: Active</div>',
                unsafe_allow_html=True
            )
        with col_status2:
            st.markdown(
                '<div class="status-active">✓ Detection: Ready</div>',
                unsafe_allow_html=True
            )


def create_input_section():
    """Create input section with file upload and camera options."""
    st.markdown("---")
    st.markdown("## 📥 Input Section")
    
    tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "🎬 Video Upload", "📹 Live Camera"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Supported formats: JPG, JPEG, PNG, BMP"
            )
        with col2:
            confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
        
        return {
            'type': 'image',
            'file': uploaded_image,
            'confidence': confidence
        }
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_video = st.file_uploader(
                "Upload a video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Supported formats: MP4, AVI, MOV, MKV"
            )
        with col2:
            confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
        
        return {
            'type': 'video',
            'file': uploaded_video,
            'confidence': confidence
        }
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            use_camera = st.checkbox("Enable Live Camera", value=False)
        with col2:
            if use_camera:
                st.markdown('<div class="status-active">✓ Camera Ready</div>', unsafe_allow_html=True)
        
        confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
        
        return {
            'type': 'camera',
            'enabled': use_camera,
            'confidence': confidence
        }


def render_detection_display(image: np.ndarray, detections: List[Dict], 
                            title: str = "Live Detection View"):
    """Render detection display with bounding boxes and plate info."""
    st.markdown(f"## {title}")
    
    # Draw detections on image
    annotated_image = image.copy()
    
    for detection in detections:
        box = detection.get('box', [])
        plate_text = detection.get('plate_text', 'UNKNOWN')
        confidence = detection.get('confidence', 0.0)
        plate_type = detection.get('plate_type', 'Unknown')
        
        if box and len(box) >= 4:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Draw bounding box
            color = get_plate_color(plate_type)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw overlay text
            label = f"{plate_text} ({confidence:.1%})"
            cv2.putText(
                annotated_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
    
    # Display image
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(annotated_image, channels='BGR', use_column_width=True)
    
    with col2:
        st.markdown("### Detection Info")
        st.metric("Detections Found", len(detections))
        if detections:
            st.markdown("**Latest Detection:**")
            latest = detections[-1]
            st.markdown(f"- **Plate:** `{latest.get('plate_text', 'N/A')}`")
            st.markdown(f"- **Type:** {latest.get('plate_type', 'N/A')}")
            st.markdown(f"- **Confidence:** {latest.get('confidence', 0):.1%}")


def create_vehicle_info_panel(detection: Dict):
    """Create card-style vehicle information panel."""
    st.markdown("## 🚙 Vehicle Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plate Number Display
        plate_text = detection.get('plate_text', 'UNKNOWN')
        plate_type = detection.get('plate_type', 'Personal')
        
        # Create styled plate number display
        plate_color_class = f"plate-{get_plate_type_color(plate_type)}"
        st.markdown(
            f'<div class="plate-number {plate_color_class}">{plate_text}</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("### Recognition Details")
        st.metric("Confidence Score", f"{detection.get('confidence', 0):.1%}")
        st.metric("Detection Time", datetime.now().strftime("%H:%M:%S"))
    
    st.markdown("---")
    
    # Vehicle details grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Plate Information")
        st.markdown(f"**Plate Type:** {detection.get('plate_type', 'N/A')}")
        st.markdown(f"**Plate Color:** {detection.get('plate_color', 'N/A')}")
        st.markdown(f"**State Code:** {detection.get('state_code', 'N/A')}")
    
    with col2:
        st.markdown("### Vehicle Details")
        st.markdown(f"**Category:** {detection.get('vehicle_category', 'N/A')}")
        st.markdown(f"**Owner:** {detection.get('owner_name', 'Not Available')}")
        st.markdown(f"**Registration:** {detection.get('registration_date', 'N/A')}")
    
    with col3:
        st.markdown("### Detection Metadata")
        st.markdown(f"**Detection Time:** {detection.get('detection_time', 'N/A')}")
        st.markdown(f"**Frame Number:** {detection.get('frame_number', 'N/A')}")
        st.markdown(f"**Processing Time:** {detection.get('processing_time', 'N/A')}ms")


def create_logs_table(history: List[Dict], page_size: int = 10):
    """Create scrollable logs table with export functionality."""
    st.markdown("## 📋 Detection History")
    
    if not history:
        st.info("No detections recorded yet. Upload an image or enable camera to start.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    df['Detection Time'] = pd.to_datetime(df.get('detection_time', datetime.now()))
    df = df.sort_values('Detection Time', ascending=False)
    
    # Display controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        entries = st.selectbox("Show entries", [5, 10, 20, 50])
    
    with col2:
        search_plate = st.text_input("Search by plate number", "")
    
    with col3:
        if st.button("📥 Export as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"alpr_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Filter data
    if search_plate:
        df = df[df.get('plate_text', '').str.contains(search_plate, case=False, na=False)]
    
    # Display table
    df_display = df.head(entries)[['plate_text', 'plate_type', 'owner_name', 
                                    'Detection Time', 'confidence']]
    df_display.columns = ['Plate Number', 'Type', 'Owner', 'Time', 'Confidence']
    
    st.dataframe(df_display, use_container_width=True)


def render_error_feedback(error_type: str, message: str = ""):
    """Render user-friendly error and debug feedback."""
    if error_type == 'no_plate':
        st.warning("⚠️ No license plate detected in the image. Try a clearer image with a visible plate.")
    elif error_type == 'low_confidence':
        st.warning(f"⚠️ Low confidence detection ({message}). Results may be inaccurate.")
    elif error_type == 'ocr_failure':
        st.error("❌ OCR processing failed. Could not extract plate text. Try a clearer image.")
    elif error_type == 'camera_error':
        st.error(f"❌ Camera error: {message}")
    elif error_type == 'processing_error':
        st.error(f"❌ Processing error: {message}")
    else:
        st.info(f"ℹ️ {message}")


def get_plate_type_color(plate_type: str) -> str:
    """Get color class for plate type."""
    plate_type = str(plate_type).lower()
    if 'commercial' in plate_type or 'red' in plate_type:
        return 'red'
    elif 'government' in plate_type or 'green' in plate_type:
        return 'green'
    else:
        return 'blue'  # Personal


def get_plate_color(plate_type: str) -> Tuple[int, int, int]:
    """Get BGR color tuple for plate type."""
    plate_type = str(plate_type).lower()
    if 'commercial' in plate_type or 'red' in plate_type:
        return (0, 0, 255)  # Red
    elif 'government' in plate_type or 'green' in plate_type:
        return (0, 255, 0)  # Green
    else:
        return (255, 0, 0)  # Blue


def create_statistics_dashboard(history: List[Dict]):
    """Create statistics and analytics dashboard."""
    st.markdown("## 📊 Analytics Dashboard")
    
    if not history:
        st.info("No data available for analytics.")
        return
    
    df = pd.DataFrame(history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Detections",
            len(df),
            delta=f"+{len(df)}" if len(df) > 0 else "0"
        )
    
    with col2:
        personal = len(df[df.get('plate_type', '').str.contains('Personal', case=False, na=False)])
        st.metric("Personal Plates", personal)
    
    with col3:
        commercial = len(df[df.get('plate_type', '').str.contains('Commercial', case=False, na=False)])
        st.metric("Commercial Plates", commercial)
    
    with col4:
        government = len(df[df.get('plate_type', '').str.contains('Government', case=False, na=False)])
        st.metric("Government Plates", government)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Plate type distribution
        plate_types = df.get('plate_type', 'Unknown').value_counts()
        fig = px.pie(
            values=plate_types.values,
            names=plate_types.index,
            title="Plate Type Distribution",
            color_discrete_sequence=[COLORS['personal_blue'], COLORS['commercial_red'], COLORS['government_green']]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        confidences = df.get('confidence', 0.0)
        fig = px.histogram(
            x=confidences,
            nbins=20,
            title="Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Count'}
        )
        fig.update_traces(marker_color=COLORS['primary'])
        st.plotly_chart(fig, use_container_width=True)


def create_sidebar_controls():
    """Create sidebar with system controls and settings."""
    with st.sidebar:
        st.markdown("### ⚙️ System Settings")
        
        # Model settings
        st.markdown("**Detection Model**")
        model_select = st.selectbox(
            "Select Model",
            ["YOLOv8 Nano", "YOLOv8 Small", "YOLOv8 Medium"],
            label_visibility="collapsed"
        )
        
        device_select = st.selectbox(
            "Device",
            ["GPU (CUDA)", "CPU"],
            label_visibility="collapsed"
        )
        
        # Processing settings
        st.markdown("---")
        st.markdown("**Processing**")
        skip_frames = st.slider("Skip Frames", 0, 10, 2)
        
        # Logging settings
        st.markdown("---")
        st.markdown("**Logging**")
        enable_logging = st.checkbox("Enable Logging", value=True)
        log_level = st.selectbox(
            "Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            label_visibility="collapsed" if enable_logging else "visible"
        )
        
        # About
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "**Nigerian ALPR System v2.0**\n\n"
            "An AI-powered automatic license plate recognition system "
            "designed for Nigerian vehicle identification and monitoring.\n\n"
            "**Tech Stack:**\n"
            "- YOLOv8 for detection\n"
            "- EasyOCR for text recognition\n"
            "- Streamlit for UI\n"
            "- Python 3.9+"
        )
        
        return {
            'model': model_select,
            'device': device_select,
            'skip_frames': skip_frames,
            'enable_logging': enable_logging,
            'log_level': log_level
        }


__all__ = [
    'apply_custom_theme',
    'render_header',
    'create_input_section',
    'render_detection_display',
    'create_vehicle_info_panel',
    'create_logs_table',
    'render_error_feedback',
    'get_plate_type_color',
    'get_plate_color',
    'create_statistics_dashboard',
    'create_sidebar_controls',
    'COLORS',
]
