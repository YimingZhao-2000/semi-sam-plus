#!/usr/bin/env python3
"""
MedSAM2 and MedSAM ViT Installation Script
Updated to use correct huggingface_hub API
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def check_git_lfs():
    """Check if git-lfs is installed."""
    try:
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
        print("✓ git-lfs is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ git-lfs is not installed")
        print("Please install git-lfs first:")
        print("  Ubuntu/Debian: sudo apt-get install git-lfs")
        print("  CentOS/RHEL: sudo yum install git-lfs")
        print("  macOS: brew install git-lfs")
        print("Then run: git lfs install")
        return False

def download_medsam_vit():
    """Download MedSAM ViT using snapshot_download."""
    print("\nDownloading MedSAM ViT Base...")
    
    local_dir = "yiming_models_hgf/medsam-vit-base"
    
    try:
        # Use snapshot_download to get the entire repository
        downloaded_path = snapshot_download(
            repo_id="wanglab/medsam-vit-base",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ MedSAM ViT downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"✗ Failed to download MedSAM ViT: {e}")
        return None

def download_medsam2():
    """Download MedSAM2 using hf_hub_download for individual files."""
    print("\nDownloading MedSAM2...")
    
    local_dir = "yiming_models_hgf/MedSAM2"
    os.makedirs(local_dir, exist_ok=True)
    
    # List of MedSAM2 checkpoint files to download
    checkpoint_files = [
        "MedSAM2_latest.pt",
        "MedSAM2_2411.pt",
        "MedSAM2_US_Heart.pt", 
        "MedSAM2_MRI_LiverLesion.pt",
        "MedSAM2_CTLesion.pt"
    ]
    
    # Additional files
    additional_files = [
        "README.md",
        "config.json"
    ]
    
    downloaded_files = []
    
    try:
        # Download checkpoint files
        for filename in checkpoint_files:
            print(f"  Downloading {filename}...")
            try:
                file_path = hf_hub_download(
                    repo_id="wanglab/MedSAM2",
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                downloaded_files.append(filename)
                print(f"    ✓ {filename} downloaded")
            except Exception as e:
                print(f"    ✗ Failed to download {filename}: {e}")
        
        # Download additional files
        for filename in additional_files:
            print(f"  Downloading {filename}...")
            try:
                file_path = hf_hub_download(
                    repo_id="wanglab/MedSAM2",
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                downloaded_files.append(filename)
                print(f"    ✓ {filename} downloaded")
            except Exception as e:
                print(f"    ✗ Failed to download {filename}: {e}")
        
        if downloaded_files:
            print(f"✓ MedSAM2 downloaded {len(downloaded_files)} files to: {local_dir}")
            return local_dir
        else:
            print("✗ No MedSAM2 files were downloaded")
            return None
            
    except Exception as e:
        print(f"✗ Failed to download MedSAM2: {e}")
        return None

def verify_model_files(model_path, model_name, expected_files):
    """Verify that required model files exist."""
    print(f"\nVerifying {model_name} files...")
    
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({file_size:,} bytes)")
        else:
            missing_files.append(file)
            print(f"  ✗ {file} missing")
    
    if missing_files:
        print(f"⚠️  Missing files for {model_name}: {missing_files}")
        return False
    else:
        print(f"✓ All {model_name} files verified")
        return True

def main():
    """Main installation function."""
    print("MedSAM2 and MedSAM ViT Installation Script")
    print("=" * 50)
    
    # Check git-lfs
    if not check_git_lfs():
        sys.exit(1)
    
    # Create base directory
    os.makedirs("yiming_models_hgf", exist_ok=True)
    
    # Download models
    installed_models = {}
    
    # Download MedSAM ViT
    medsam_vit_path = download_medsam_vit()
    if medsam_vit_path:
        installed_models['MedSAM ViT Base'] = medsam_vit_path
        # Verify MedSAM ViT files
        verify_model_files(
            medsam_vit_path, 
            "MedSAM ViT Base", 
            ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        )
    
    # Download MedSAM2
    medsam2_path = download_medsam2()
    if medsam2_path:
        installed_models['MedSAM2'] = medsam2_path
        # Verify MedSAM2 files
        verify_model_files(
            medsam2_path,
            "MedSAM2",
            ['MedSAM2_latest.pt', 'README.md']
        )
    
    # Summary
    print(f"\n{'='*50}")
    print("INSTALLATION SUMMARY")
    print(f"{'='*50}")
    print(f"Base directory: {Path('yiming_models_hgf').absolute()}")
    
    for name, path in installed_models.items():
        print(f"{name}: {path}")
    
    if not installed_models:
        print("⚠️  No models were successfully installed")
        print("Please check your internet connection and try again")
        sys.exit(1)
    
    print(f"\n✓ Installation complete! {len(installed_models)} model(s) installed")
    print("\nNext steps:")
    print("1. Install dependencies: pip install transformers torch")
    print("2. Install MedSAM2 package: pip install git+https://github.com/bowang-lab/MedSAM2.git")
    print("3. Test setup: python test_model_loading.py")

if __name__ == "__main__":
    main() 