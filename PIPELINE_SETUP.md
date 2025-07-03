# Semi-SAM+ 3D Medical Image Segmentation Pipeline Setup

## Overview

This pipeline implements a semi-supervised learning approach for 3D medical image segmentation using:
- **Student Model**: SlicingSAM3D (custom 3D decoder with MedSAM ViT as frozen encoder)
- **Teacher Models**: MedSAM2 (3D) and MedSAM ViT (2D) for pseudo-label generation

## Model Architecture

### Student Model: SlicingSAM3D
```
Input: [B, 1, D, H, W] 3D volume
├── Slice along D dimension → [B, 1, H, W] 2D slices
├── MedSAM ViT Encoder → [B, 256, H//16, W//16] features per slice
├── Stack features → [B, 256, D, H//16, W//16] 3D features
└── 3D Decoder (4 layers) → [B, 1, D, H, W] segmentation mask
```

### Teacher Models
1. **MedSAM ViT**: 2D SAM model for slice-by-slice processing
2. **MedSAM2**: 3D model for volume-level processing with prompts

## Pipeline Integration

### Model Loading
- **MedSAM ViT**: Loads from `./yiming_models_hgf/medsam-vit-base/`
- **MedSAM2**: Loads from `./yiming_models_hgf/MedSAM2/MedSAM2_latest.pt`

### Pseudo-label Generation
- **MedSAM ViT**: Processes each slice independently without prompts
- **MedSAM2**: Uses point and mask prompts for 3D volume processing

## Setup Instructions

### 1. Install Models
```bash
# Install with all files (recommended)
python install_models.py

# Install with LFS files skipped (faster initial setup)
python install_models.py --skip-lfs
```

### 2. Install Dependencies
```bash
# Core dependencies
pip install torch torchvision transformers

# MedSAM2 package
pip install git+https://github.com/bowang-lab/MedSAM2.git

# Optional: Medical image processing
pip install torchio
```

### 3. Download LFS Files (if skipped)
```bash
# Download MedSAM ViT files
cd yiming_models_hgf/medsam-vit-base && git lfs pull

# Download MedSAM2 checkpoint files
cd yiming_models_hgf/MedSAM2 && git lfs pull
```

### 4. Test Setup
```bash
python test_model_loading.py
```

## Configuration

### Model Selection
```python
# In config.py
student_model = 'slicing-sam3d'  # Custom 3D model
teacher_models = 'medsam2'       # or 'medsam-vit'

# Model paths
medsam_vit_path = './yiming_models_hgf/medsam-vit-base'
medsam2_path = './yiming_models_hgf/MedSAM2'
```

### Training Phases
1. **Warmup**: Train student model on labeled data only
2. **Semi-supervised**: Use teacher models for pseudo-label generation
3. **Fine-tune**: Final training on all data

## Usage Examples

### Basic Training
```python
from config import Config
from model import get_student_model, get_teacher_model
from train import train_semi

# Load configuration
config = Config()

# Load models
student = get_student_model(config)
teacher = get_teacher_model(config, 'medsam2')

# Train
train_semi(config, student, teacher, optimizer, train_loader, unlabeled_loader, device)
```

### Model Loading Functions
```python
from model import get_medsam_vit_model, get_medsam2_model

# Load MedSAM ViT
processor, model = get_medsam_vit_model(device='cuda')

# Load MedSAM2
model = get_medsam2_model(device='cuda')
```

## Key Features

### 1. Proper Model Detection
- Automatically detects model type based on class name
- Handles both MedSAM ViT and MedSAM2 APIs correctly

### 2. Error Handling
- Checks for missing model files before attempting to load
- Provides clear error messages and installation instructions
- Graceful fallback for missing dependencies

### 3. Efficient Loading
- No automatic downloading during training
- Uses local model files only
- Proper file existence checks

### 4. Flexible Architecture
- Supports multiple teacher models
- Easy to extend with new models
- Configurable model parameters

## Troubleshooting

### Common Issues

1. **Missing Model Files**
   ```
   FileNotFoundError: MedSAM ViT model not found
   ```
   **Solution**: Run `python install_models.py`

2. **Missing LFS Files**
   ```
   FileNotFoundError: pytorch_model.bin not found
   ```
   **Solution**: Run `git lfs pull` in model directory

3. **Missing MedSAM2 Package**
   ```
   ImportError: medsam2 package is required
   ```
   **Solution**: `pip install git+https://github.com/bowang-lab/MedSAM2.git`

4. **Model Loading Errors**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use CPU for testing

### Testing
Run the test script to diagnose issues:
```bash
python test_model_loading.py
```

## File Structure
```
model/
├── model.py              # Model definitions and loading
├── config.py             # Configuration settings
├── train.py              # Training loops
├── utils.py              # Utility functions
├── test_model_loading.py # Setup verification
├── install_models.py     # Model installation
└── yiming_models_hgf/    # Local model storage
    ├── medsam-vit-base/  # MedSAM ViT files
    └── MedSAM2/          # MedSAM2 files
```

## Next Steps

1. **Data Loading**: Implement data loaders for your medical image dataset
2. **Training Script**: Set up the main training script with your data
3. **Evaluation**: Add evaluation metrics and validation
4. **Inference**: Create inference scripts for new data

The pipeline is now properly set up to handle both MedSAM ViT and MedSAM2 models with correct error handling and efficient loading. 