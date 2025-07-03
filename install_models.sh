#!/bin/bash

# MedSAM2 and MedSAM ViT Installation Script
# Updated to use correct huggingface_hub API

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_git_lfs() {
    print_info "Checking git-lfs installation..."
    if command -v git-lfs &> /dev/null; then
        print_success "git-lfs is installed"
        return 0
    else
        print_error "git-lfs is not installed"
        echo "Please install git-lfs first:"
        echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
        echo "  CentOS/RHEL: sudo yum install git-lfs"
        echo "  macOS: brew install git-lfs"
        echo "Then run: git lfs install"
        return 1
    fi
}

check_python_deps() {
    print_info "Checking Python dependencies..."
    
    # Check if huggingface_hub is available
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        print_success "huggingface_hub is available"
    else
        print_warning "huggingface_hub not found, installing..."
        pip3 install huggingface_hub
    fi
}

download_medsam_vit() {
    print_info "Downloading MedSAM ViT Base..."
    
    local_dir="yiming_models_hgf/medsam-vit-base"
    
    # Use Python script to download
    python3 -c "
import os
from huggingface_hub import snapshot_download

try:
    downloaded_path = snapshot_download(
        repo_id='wanglab/medsam-vit-base',
        local_dir='$local_dir',
        local_dir_use_symlinks=False
    )
    print(f'✓ MedSAM ViT downloaded to: {downloaded_path}')
except Exception as e:
    print(f'✗ Failed to download MedSAM ViT: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "MedSAM ViT downloaded successfully"
        return 0
    else
        print_error "Failed to download MedSAM ViT"
        return 1
    fi
}

download_medsam2() {
    print_info "Downloading MedSAM2..."
    
    local_dir="yiming_models_hgf/MedSAM2"
    mkdir -p "$local_dir"
    
    # Use Python script to download individual files
    python3 -c "
import os
from huggingface_hub import hf_hub_download

local_dir = '$local_dir'
checkpoint_files = [
    'MedSAM2_latest.pt',
    'MedSAM2_2411.pt',
    'MedSAM2_US_Heart.pt',
    'MedSAM2_MRI_LiverLesion.pt',
    'MedSAM2_CTLesion.pt'
]
additional_files = ['README.md', 'config.json']

downloaded_count = 0

# Download checkpoint files
for filename in checkpoint_files:
    try:
        print(f'  Downloading {filename}...')
        file_path = hf_hub_download(
            repo_id='wanglab/MedSAM2',
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        downloaded_count += 1
        print(f'    ✓ {filename} downloaded')
    except Exception as e:
        print(f'    ✗ Failed to download {filename}: {e}')

# Download additional files
for filename in additional_files:
    try:
        print(f'  Downloading {filename}...')
        file_path = hf_hub_download(
            repo_id='wanglab/MedSAM2',
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        downloaded_count += 1
        print(f'    ✓ {filename} downloaded')
    except Exception as e:
        print(f'    ✗ Failed to download {filename}: {e}')

if downloaded_count > 0:
    print(f'✓ MedSAM2 downloaded {downloaded_count} files to: {local_dir}')
    exit(0)
else:
    print('✗ No MedSAM2 files were downloaded')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "MedSAM2 downloaded successfully"
        return 0
    else
        print_error "Failed to download MedSAM2"
        return 1
    fi
}

verify_files() {
    local model_path="$1"
    local model_name="$2"
    shift 2
    local expected_files=("$@")
    
    print_info "Verifying $model_name files..."
    
    missing_files=()
    for file in "${expected_files[@]}"; do
        file_path="$model_path/$file"
        if [ -f "$file_path" ]; then
            file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null || echo "unknown")
            print_success "$file ($file_size bytes)"
        else
            missing_files+=("$file")
            print_error "$file missing"
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_success "All $model_name files verified"
        return 0
    else
        print_warning "Missing files for $model_name: ${missing_files[*]}"
        return 1
    fi
}

main() {
    echo "MedSAM2 and MedSAM ViT Installation Script"
    echo "=================================================="
    
    # Check git-lfs
    if ! check_git_lfs; then
        exit 1
    fi
    
    # Check Python dependencies
    check_python_deps
    
    # Create base directory
    mkdir -p yiming_models_hgf
    
    # Download models
    installed_models=()
    
    # Download MedSAM ViT
    if download_medsam_vit; then
        installed_models+=("MedSAM ViT Base")
        verify_files "yiming_models_hgf/medsam-vit-base" "MedSAM ViT Base" \
            "config.json" "pytorch_model.bin" "tokenizer.json"
    fi
    
    # Download MedSAM2
    if download_medsam2; then
        installed_models+=("MedSAM2")
        verify_files "yiming_models_hgf/MedSAM2" "MedSAM2" \
            "MedSAM2_latest.pt" "README.md"
    fi
    
    # Summary
    echo ""
    echo "=================================================="
    echo "INSTALLATION SUMMARY"
    echo "=================================================="
    echo "Base directory: $(pwd)/yiming_models_hgf"
    
    if [ ${#installed_models[@]} -eq 0 ]; then
        print_warning "No models were successfully installed"
        echo "Please check your internet connection and try again"
        exit 1
    fi
    
    for model in "${installed_models[@]}"; do
        echo "$model: yiming_models_hgf/$model"
    done
    
    echo ""
    print_success "Installation complete! ${#installed_models[@]} model(s) installed"
    echo ""
    echo "Next steps:"
    echo "1. Install dependencies: pip install transformers torch"
    echo "2. Install MedSAM2 package: pip install git+https://github.com/bowang-lab/MedSAM2.git"
    echo "3. Test setup: python test_model_loading.py"
}

# Run main function
main "$@" 