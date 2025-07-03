#!/usr/bin/env python3
"""
Test script to verify model loading functionality.
This script checks if models can be loaded properly without downloading them.
"""

import os
import sys
from pathlib import Path

def test_model_paths():
    """Test if model paths exist and have required files."""
    print("Testing model paths...")
    
    # Check MedSAM ViT
    medsam_vit_path = './yiming_models_hgf/medsam-vit-base'
    if os.path.exists(medsam_vit_path):
        print(f"✓ MedSAM ViT path exists: {medsam_vit_path}")
        
        # Check required files
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        missing_files = []
        for file in required_files:
            file_path = os.path.join(medsam_vit_path, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file} exists")
            else:
                missing_files.append(file)
                print(f"  ✗ {file} missing")
        
        if missing_files:
            print(f"  ⚠️  Missing files: {missing_files}")
            print("  Run: cd yiming_models_hgf/medsam-vit-base && git lfs pull")
        else:
            print("  ✓ All required files present")
    else:
        print(f"✗ MedSAM ViT path missing: {medsam_vit_path}")
        print("  Run: python install_models.py")
    
    # Check MedSAM2
    medsam2_path = './yiming_models_hgf/MedSAM2'
    if os.path.exists(medsam2_path):
        print(f"✓ MedSAM2 path exists: {medsam2_path}")
        
        # Check checkpoint files
        checkpoint_files = [
            'MedSAM2_latest.pt',
            'MedSAM2_2411.pt', 
            'MedSAM2_US_Heart.pt',
            'MedSAM2_MRI_LiverLesion.pt',
            'MedSAM2_CTLesion.pt'
        ]
        
        found_checkpoints = []
        for ckpt in checkpoint_files:
            ckpt_path = os.path.join(medsam2_path, ckpt)
            if os.path.exists(ckpt_path):
                found_checkpoints.append(ckpt)
                print(f"  ✓ {ckpt} exists")
            else:
                print(f"  ✗ {ckpt} missing")
        
        if not found_checkpoints:
            print("  ⚠️  No checkpoint files found")
            print("  Run: cd yiming_models_hgf/MedSAM2 && git lfs pull")
        else:
            print(f"  ✓ Found {len(found_checkpoints)} checkpoint(s)")
    else:
        print(f"✗ MedSAM2 path missing: {medsam2_path}")
        print("  Run: python install_models.py")

def test_model_imports():
    """Test if required packages can be imported."""
    print("\nTesting model imports...")
    
    # Test transformers
    try:
        from transformers import AutoProcessor, AutoModelForMaskGeneration
        print("✓ transformers package available")
    except ImportError as e:
        print(f"✗ transformers package missing: {e}")
        print("  Run: pip install transformers")
    
    # Test MedSAM2 package
    try:
        import medsam2
        print("✓ medsam2 package available")
    except ImportError as e:
        print(f"✗ medsam2 package missing: {e}")
        print("  Run: pip install git+https://github.com/bowang-lab/MedSAM2.git")
    
    # Test other dependencies
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError as e:
        print(f"✗ PyTorch missing: {e}")
    
    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError as e:
        print(f"✗ NumPy missing: {e}")

def test_model_loading_functions():
    """Test the model loading functions from model.py."""
    print("\nTesting model loading functions...")
    
    try:
        from model import get_medsam_vit_model, get_medsam2_model
        print("✓ Model loading functions imported successfully")
        
        # Test MedSAM ViT loading (without actually loading)
        try:
            # This will fail if files are missing, which is expected
            processor, model = get_medsam_vit_model(device='cpu')
            print("✓ MedSAM ViT loading function works")
        except FileNotFoundError as e:
            print(f"⚠️  MedSAM ViT loading failed (expected if files missing): {e}")
        except Exception as e:
            print(f"✗ MedSAM ViT loading error: {e}")
        
        # Test MedSAM2 loading (without actually loading)
        try:
            # This will fail if package is missing, which is expected
            model = get_medsam2_model(device='cpu')
            print("✓ MedSAM2 loading function works")
        except ImportError as e:
            print(f"⚠️  MedSAM2 loading failed (expected if package missing): {e}")
        except Exception as e:
            print(f"✗ MedSAM2 loading error: {e}")
            
    except ImportError as e:
        print(f"✗ Failed to import model loading functions: {e}")

def test_pipeline_integration():
    """Test the pipeline integration functions."""
    print("\nTesting pipeline integration...")
    
    try:
        from model import get_student_model, get_teacher_model
        from config import Config
        
        print("✓ Pipeline functions imported successfully")
        
        # Create a test config
        config = Config()
        config.device = 'cpu'
        
        # Test student model loading
        try:
            student = get_student_model(config)
            print("✓ Student model loading works")
        except Exception as e:
            print(f"⚠️  Student model loading failed (expected if models missing): {e}")
        
        # Test teacher model loading
        try:
            teacher = get_teacher_model(config, 'medsam-vit')
            print("✓ MedSAM ViT teacher loading works")
        except Exception as e:
            print(f"⚠️  MedSAM ViT teacher loading failed (expected if models missing): {e}")
        
        try:
            teacher = get_teacher_model(config, 'medsam2')
            print("✓ MedSAM2 teacher loading works")
        except Exception as e:
            print(f"⚠️  MedSAM2 teacher loading failed (expected if models missing): {e}")
            
    except ImportError as e:
        print(f"✗ Failed to import pipeline functions: {e}")

def main():
    """Run all tests."""
    print("Semi-SAM+ Model Loading Test")
    print("=" * 50)
    
    test_model_paths()
    test_model_imports()
    test_model_loading_functions()
    test_pipeline_integration()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("- Check ✓ for working components")
    print("- Check ⚠️  for expected failures (missing models/packages)")
    print("- Check ✗ for unexpected errors")
    print("\nTo fix issues:")
    print("1. Install models: python install_models.py")
    print("2. Install packages: pip install transformers medsam2")
    print("3. Download LFS files: git lfs pull in model directories")

if __name__ == "__main__":
    main() 