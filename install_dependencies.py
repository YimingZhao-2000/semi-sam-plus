#!/usr/bin/env python3
"""
Dependency Installation Script for Semi-SAM+ Pipeline
Focuses on installing required packages for online model loading
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    if description:
        print(f"  {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"  ✓ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        if e.stderr:
            print(f"    Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_core_dependencies():
    """Install core PyTorch and transformers dependencies."""
    print("\nInstalling core dependencies...")
    
    # Install PyTorch (adjust CUDA version as needed)
    if not run_command("pip install torch torchvision torchaudio", "Installing PyTorch"):
        print("⚠️  PyTorch installation failed. You may need to install it manually.")
    
    # Install transformers and huggingface_hub
    if not run_command("pip install transformers huggingface_hub", "Installing transformers"):
        return False
    
    # Install other required packages
    packages = [
        "numpy",
        "scipy", 
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "pillow",
        "opencv-python",
        "pandas"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}")
    
    return True

def install_medical_dependencies():
    """Install medical image processing dependencies."""
    print("\nInstalling medical image processing dependencies...")
    
    medical_packages = [
        "torchio",
        "nibabel", 
        "SimpleITK"
    ]
    
    for package in medical_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}")
    
    return True

def install_medsam2():
    """Install MedSAM2 package."""
    print("\nInstalling MedSAM2 package...")
    
    if not run_command("pip install git+https://github.com/bowang-lab/MedSAM2.git", "Installing MedSAM2"):
        print("⚠️  MedSAM2 installation failed. You may need to install it manually.")
        print("   Try: pip install git+https://github.com/bowang-lab/MedSAM2.git")
        return False
    
    return True

def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting imports...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("numpy", "NumPy"),
        ("torchio", "TorchIO"),
        ("nibabel", "Nibabel"),
        ("SimpleITK", "SimpleITK")
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"  ✓ {name} imported successfully")
        except ImportError as e:
            print(f"  ✗ {name} import failed: {e}")
            failed_imports.append(name)
    
    # Test MedSAM2 import
    try:
        import medsam2
        print("  ✓ MedSAM2 imported successfully")
    except ImportError as e:
        print(f"  ✗ MedSAM2 import failed: {e}")
        failed_imports.append("MedSAM2")
    
    return len(failed_imports) == 0, failed_imports

def test_model_loading():
    """Test if models can be loaded from HuggingFace Hub."""
    print("\nTesting model loading from HuggingFace Hub...")
    
    try:
        from transformers import AutoProcessor, AutoModelForMaskGeneration
        
        print("  Testing MedSAM ViT loading...")
        processor = AutoProcessor.from_pretrained("wanglab/medsam-vit-base")
        model = AutoModelForMaskGeneration.from_pretrained("wanglab/medsam-vit-base")
        print("  ✓ MedSAM ViT loaded successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False

def main():
    """Main installation function."""
    print("Semi-SAM+ Dependency Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_core_dependencies():
        print("✗ Core dependency installation failed")
        sys.exit(1)
    
    if not install_medical_dependencies():
        print("⚠️  Some medical dependencies failed to install")
    
    if not install_medsam2():
        print("⚠️  MedSAM2 installation failed")
    
    # Test imports
    imports_ok, failed_imports = test_imports()
    
    # Test model loading
    model_loading_ok = test_model_loading()
    
    # Summary
    print(f"\n{'='*50}")
    print("INSTALLATION SUMMARY")
    print(f"{'='*50}")
    
    if imports_ok and model_loading_ok:
        print("✓ All dependencies installed and working!")
        print("✓ Models can be loaded from HuggingFace Hub")
        print("\nYou can now use the pipeline with online model loading:")
        print("  from model import get_medsam_vit_model, get_medsam2_model")
        print("  processor, model = get_medsam_vit_model(device='cuda')")
    else:
        print("⚠️  Some issues detected:")
        if not imports_ok:
            print(f"  - Failed imports: {', '.join(failed_imports)}")
        if not model_loading_ok:
            print("  - Model loading from HuggingFace Hub failed")
        print("\nYou may need to:")
        print("  1. Check your internet connection")
        print("  2. Install missing packages manually")
        print("  3. Use local model installation as fallback")

if __name__ == "__main__":
    main() 