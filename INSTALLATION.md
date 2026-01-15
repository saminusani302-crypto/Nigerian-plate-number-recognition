# Installation and Setup Guide - Nigerian ALPR System

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment support
- Git (for cloning repository)
- CUDA 11.8+ (optional, for GPU support)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Nigerian-plate-number-recognition.git
cd Nigerian-plate-number-recognition
```

### 2. Create Virtual Environment

**On Linux/MacOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Upgrade pip, setuptools, and wheel

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **YOLOv8**: Object detection model
- **EasyOCR**: Optical character recognition
- **Flask**: Web framework
- **NumPy, Pandas**: Data processing

### 5. Verify Installation

Run the quickstart script:

```bash
python quickstart.py
```

Expected output:
```
╔════════════════════════════════════════════════════════════╗
║   Nigerian Automatic License Plate Recognition System      ║
║                      ALPR v1.0.0                           ║
║                                                            ║
║   Powered by YOLOv8 & EasyOCR                             ║
╚════════════════════════════════════════════════════════════╝

🔧 Initializing ALPR System...
✓ ALPR Pipeline initialized successfully
✓ Detection Device: cpu
✓ Model: YOLOv8
✓ Input Size: [640, 640]
✓ Classes: 80 detected
...
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Support

For faster processing, use NVIDIA GPU with CUDA:

#### 1. Check NVIDIA GPU

```bash
nvidia-smi
```

#### 2. Install CUDA-enabled PyTorch

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Verify GPU Setup

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### Run ALPR with GPU

```python
from alpr_system import ALPRPipeline

alpr = ALPRPipeline(device='cuda')  # Use GPU
```

## Starting the Application

### Method 1: Web Interface (Recommended)

```bash
python app.py
```

Then open your browser:
```
http://localhost:5000
```

### Method 2: Python Script

```python
from alpr_system import ALPRPipeline

alpr = ALPRPipeline()

# Process image
results = alpr.process_image('image.jpg')
print(f"Plates detected: {results['plates']}")

# Process video
stats = alpr.process_video('video.mp4', display=True)
print(f"Unique plates: {stats['unique_plates']}")
```

### Method 3: Command Line

```bash
python -c "from alpr_system import ALPRPipeline; alpr = ALPRPipeline(); print(alpr.get_statistics())"
```

## Configuration

### 1. Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Flask Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# ALPR Configuration
MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
DEVICE=cpu  # Set to 'cuda' for GPU

# Logging
LOG_DIR=logs
LOG_FORMAT=csv
```

### 2. Model Selection

Choose YOLOv8 variant for your needs:

| Model | Size | Speed | Memory | Accuracy | Use Case |
|-------|------|-------|--------|----------|----------|
| yolov8n.pt | 12MB | ⚡⚡⚡ | Low | Good | Edge devices |
| yolov8s.pt | 45MB | ⚡⚡ | Medium | Better | Balanced |
| yolov8m.pt | 49MB | ⚡ | High | Best | High accuracy |

## Testing

Run the test suite:

```bash
python test_alpr.py
```

This will run:
- Preprocessor tests
- Detector tests
- OCR tests
- Logger tests
- Pipeline tests
- Integration tests

## File Structure

```
Nigerian-plate-number-recognition/
├── alpr_system/                    # Main package
│   ├── __init__.py
│   ├── main.py                     # ALPRPipeline class
│   ├── detector.py                 # YOLOv8 detection
│   ├── ocr.py                      # EasyOCR
│   ├── preprocessor.py             # Image preprocessing
│   └── logger.py                   # Results logging
├── models/                         # Model weights
├── dataset/                        # Training data
├── logs/                           # Detection logs
├── static/                         # Web assets
├── app.py                          # Flask app
├── index.html                      # Web UI
├── requirements.txt                # Dependencies
├── quickstart.py                   # Quick test script
├── test_alpr.py                    # Unit tests
├── .env.example                    # Config template
└── README.md                       # Documentation
```

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:**
```bash
pip install ultralytics>=8.0.0
```

### Issue: "CUDA out of memory"

**Solution:**
```python
# Use CPU instead
alpr = ALPRPipeline(device='cpu')

# Or use smaller model
alpr = ALPRPipeline(model_path='yolov8n.pt')
```

### Issue: "FileNotFoundError: weights/yolov8n.pt not found"

**Solution:**
The model will be auto-downloaded. Ensure internet connection and writable `~/.cache/` directory.

### Issue: "ImportError: cannot import name 'YOLO'"

**Solution:**
```bash
pip install --upgrade ultralytics torch
```

### Issue: Webcam not detected

**Solution:**
```bash
# Check available cameras
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i} available')
        cap.release()
"
```

## Performance Optimization

### 1. Use GPU Acceleration
- 3-5x faster than CPU
- Requires NVIDIA GPU and CUDA

### 2. Reduce Resolution
- Process at lower resolution if possible
- Trade-off between speed and accuracy

### 3. Use Smaller Models
- YOLOv8n: fastest, lowest memory
- YOLOv8m: slowest, highest accuracy

### 4. Skip Frames
- Process every Nth frame in videos
- Trade-off for real-time performance

### 5. Adjust Confidence Threshold
```python
alpr.detector.set_confidence_threshold(0.6)  # Higher = fewer detections, faster
```

## Updating the System

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Update Models

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"  # Downloads latest
```

## Uninstall

### Remove Virtual Environment

```bash
deactivate  # Exit virtual environment
rm -rf venv  # Remove directory
```

### Clean Cache

```bash
rm -rf ~/.cache/torch
rm -rf ~/.cache/ultralytics
rm -rf ~/.cache/easyocr
```

## Docker Setup (Optional)

### Build Docker Image

```bash
docker build -t nigerian-alpr:1.0 .
```

### Run Docker Container

```bash
docker run -p 5000:5000 -v $(pwd)/logs:/app/logs nigerian-alpr:1.0
```

## Next Steps

1. **Web Interface**: Open `http://localhost:5000`
2. **Upload Images**: Test with vehicle images
3. **Start Webcam**: Real-time plate detection
4. **View Logs**: Check detected plates
5. **Export Results**: Download CSV/JSON files

## Support

- **Documentation**: See README.md
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [CUDA Installation](https://developer.nvidia.com/cuda-downloads)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Version**: 1.0.0  
**Last Updated**: January 2024
