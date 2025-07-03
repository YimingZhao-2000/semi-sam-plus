# Model Download Documentation

This document explains how to download and install MedSAM2 and MedSAM ViT models for the Semi-SAM+ 3D Medical Image Segmentation Pipeline.

## Prerequisites

### 1. Install git-lfs on Linux Server

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install git-lfs

# CentOS/RHEL
sudo yum install git-lfs

# Initialize git-lfs
git lfs install
```

### 2. Install Python Dependencies

```bash
# Install huggingface_hub for model downloading
pip install huggingface_hub

# Install other required packages
pip install torch torchvision transformers
```

### 3. Verify git-lfs installation

```bash
git lfs version
# Should show: git-lfs/x.x.x (GitHub; linux amd64; go x.x.x)
```

## Installation Methods

### Method 1: Using Python Script (Recommended)

```bash
# Make script executable
chmod +x install_models.py

# Install models using correct huggingface_hub API
python3 install_models.py
```

### Method 2: Using Bash Script

```bash
# Make script executable
chmod +x install_models.sh

# Install models
./install_models.sh
```

## What Gets Downloaded

### Directory Structure
```
yiming_models_hgf/
├── MedSAM2/
│   ├── MedSAM2_latest.pt    # Latest MedSAM2 checkpoint
│   ├── MedSAM2_2411.pt      # Base model checkpoint
│   ├── MedSAM2_US_Heart.pt  # Heart ultrasound model
│   ├── MedSAM2_MRI_LiverLesion.pt  # Liver lesion model
│   ├── MedSAM2_CTLesion.pt  # CT lesion model
│   ├── README.md            # Model documentation
│   └── config.json          # Model configuration
└── medsam-vit-base/
    ├── config.json          # Model configuration
    ├── pytorch_model.bin    # Model weights
    ├── tokenizer.json       # Tokenizer
    ├── preprocessor_config.json  # Preprocessor config
    └── ... (other HuggingFace files)
```

### Model Details

#### MedSAM2
- **Repository**: https://huggingface.co/wanglab/MedSAM2
- **Checkpoint**: MedSAM2_latest.pt (recommended)
- **Size**: ~600MB total for all checkpoints
- **Usage**: Teacher model for pseudo-label generation
- **Download Method**: Individual files using `hf_hub_download`

#### MedSAM ViT Base
- **Repository**: https://huggingface.co/wanglab/medsam-vit-base
- **Model**: SAM ViT-B encoder
- **Size**: ~88MB
- **Usage**: 2D encoder for Slicing SAM3D student model
- **Download Method**: Complete repository using `snapshot_download`

## Installation Process

### 1. MedSAM ViT Installation
The script uses `snapshot_download` to download the entire MedSAM ViT repository:

```python
from huggingface_hub import snapshot_download

downloaded_path = snapshot_download(
    repo_id="wanglab/medsam-vit-base",
    local_dir="yiming_models_hgf/medsam-vit-base",
    local_dir_use_symlinks=False
)
```

### 2. MedSAM2 Installation
The script uses `hf_hub_download` to download individual checkpoint files:

```python
from huggingface_hub import hf_hub_download

# Download each checkpoint file individually
for filename in checkpoint_files:
    file_path = hf_hub_download(
        repo_id="wanglab/MedSAM2",
        filename=filename,  # Single filename, not filenames
        local_dir="yiming_models_hgf/MedSAM2",
        local_dir_use_symlinks=False
    )
```

## Installation Options

### Automatic Installation
The scripts automatically:
- Check for git-lfs installation
- Install huggingface_hub if missing
- Download all model files
- Verify file integrity
- Provide detailed progress feedback

### Manual Installation
If you prefer manual installation:

```python
# Install MedSAM ViT
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="wanglab/medsam-vit-base",
    local_dir="yiming_models_hgf/medsam-vit-base"
)

# Install MedSAM2 checkpoints
from huggingface_hub import hf_hub_download
checkpoints = ["MedSAM2_latest.pt", "MedSAM2_2411.pt", ...]
for ckpt in checkpoints:
    hf_hub_download(
        repo_id="wanglab/MedSAM2",
        filename=ckpt,
        local_dir="yiming_models_hgf/MedSAM2"
    )
```

## Troubleshooting

### Common Issues

1. **huggingface_hub Import Error**
   ```
   ModuleNotFoundError: No module named 'huggingface_hub'
   ```
   **Solution**: `pip install huggingface_hub`

2. **git-lfs Not Found**
   ```
   git: 'lfs' is not a git command
   ```
   **Solution**: Install git-lfs using your package manager

3. **Download Failures**
   ```
   ConnectionError: Failed to download file
   ```
   **Solution**: Check internet connection and try again

4. **Permission Errors**
   ```
   Permission denied: yiming_models_hgf/
   ```
   **Solution**: Ensure write permissions in current directory

### Verification

After installation, verify the setup:

```bash
# Check file sizes (should be large, not just pointers)
ls -lh yiming_models_hgf/medsam-vit-base/pytorch_model.bin
ls -lh yiming_models_hgf/MedSAM2/MedSAM2_latest.pt

# Run test script
python test_model_loading.py
```

## API Corrections

### Important Notes
- Use `filename` (singular) not `filenames` (plural) with `hf_hub_download`
- Use `snapshot_download` for complete repositories
- Use `hf_hub_download` for individual files
- Always specify `local_dir_use_symlinks=False` for actual file downloads

### Correct Usage Examples

```python
# ✅ Correct: Single file download
hf_hub_download(
    repo_id="wanglab/MedSAM2",
    filename="MedSAM2_latest.pt",  # Single filename
    local_dir="models"
)

# ✅ Correct: Complete repository download
snapshot_download(
    repo_id="wanglab/medsam-vit-base",
    local_dir="models/medsam-vit"
)

# ❌ Incorrect: Multiple files with filenames
hf_hub_download(
    repo_id="wanglab/MedSAM2",
    filenames=["a.pt", "b.pt"]  # This will fail
)
```

## Next Steps

After successful installation:

1. **Install MedSAM2 Package**:
   ```bash
   pip install git+https://github.com/bowang-lab/MedSAM2.git
   ```

2. **Test Model Loading**:
   ```bash
   python test_model_loading.py
   ```

3. **Start Training**:
   ```python
   from model import get_student_model, get_teacher_model
   # Your training code here
   ```

The installation scripts now use the correct huggingface_hub API and provide robust error handling for reliable model downloads. 