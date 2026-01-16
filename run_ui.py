
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║        🚗 Nigerian ALPR System - Professional UI v2.0 🚗       ║
║                                                                ║
║     Automatic License Plate Recognition & Vehicle Tracking    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n📦 Checking dependencies...\n")
    
    required_packages = [
        'streamlit',
        'opencv-python',
        'torch',
        'ultralytics',
        'easyocr',
        'pandas',
        'numpy',
        'plotly',
        'pillow',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("\n📥 Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✓ Dependencies installed!")
    else:
        print("\n✓ All dependencies are installed!")
    
    return len(missing_packages) == 0


def check_models():
    """Check if pre-trained models are available."""
    print("\n🤖 Checking models...\n")
    
    model_path = Path('yolov8n.pt')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ YOLOv8 Nano model found ({size_mb:.1f} MB)")
    else:
        print("  ℹ️  YOLOv8 Nano will auto-download on first run (~100 MB)")
    
    print("  ℹ️  EasyOCR models will auto-download on first run (~100 MB)")


def print_next_steps():
    """Print next steps for running the UI."""
    print("\n" + "="*60)
    print("🚀 QUICK START")
    print("="*60)
    print("\n✨ To launch the Nigerian ALPR System UI:\n")
    print("  1️⃣  Make sure you're in the project directory:")
    print("     cd Nigerian-plate-number-recognition\n")
    print("  2️⃣  Run Streamlit UI (Recommended):")
    print("     streamlit run alpr_system/ui/app.py\n")
    print("  3️⃣  Open your browser and navigate to:")
    print("     http://localhost:8501\n")
    print("="*60)
    print("\n📚 For detailed documentation, see: STREAMLIT_UI_README.md\n")


def print_system_info():
    """Print system information."""
    print("\n💻 System Information:")
    print(f"  - Python version: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  - PyTorch: not installed")
    
    try:
        import streamlit
        print(f"  - Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("  - Streamlit: not installed")
    
    print(f"  - Platform: {sys.platform}")


def main():
    """Main entry point."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return False
    
    # Check models
    check_models()
    
    # System info
    print_system_info()
    
    # Next steps
    print_next_steps()
    
    # Ask user if they want to start
    print("🎬 Would you like to start the UI now? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\n🚀 Starting Streamlit UI...\n")
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "alpr_system/ui/app.py"
            ])
        except KeyboardInterrupt:
            print("\n\n👋 Streamlit UI stopped.")
        except Exception as e:
            print(f"\n❌ Error starting UI: {e}")
            return False
    else:
        print("\n👋 You can start the UI later with:")
        print("   streamlit run alpr_system/ui/app.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
