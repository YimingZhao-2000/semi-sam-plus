# Semi-SAM+ 3D Medical Image Segmentation Pipeline

A comprehensive semi-supervised learning pipeline for 3D medical image segmentation using MedSAM ViT and MedSAM2 models with **online model loading**.

## 🚀 Features

- **Student Model**: Custom SlicingSAM3D with MedSAM ViT as frozen encoder
- **Teacher Models**: MedSAM2 (3D) and MedSAM ViT (2D) for pseudo-label generation
- **Semi-supervised Training**: Efficient training with labeled and unlabeled data
- **Online Model Loading**: Models loaded directly from HuggingFace Hub
- **Robust Fallback**: Automatic fallback to local models if online loading fails
- **Flexible Architecture**: Easy to extend with new models

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Internet connection for model downloading

### Dependencies
```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
huggingface_hub>=0.15.0

# Medical image processing
torchio>=0.18.0
nibabel>=5.0.0
SimpleITK>=2.0.0

# Scientific computing
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0

# Utilities
tqdm>=4.64.0
pillow>=9.0.0
opencv-python>=4.6.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YimingZhao-2000/semi-sam-plus.git
cd semi-sam-plus
```

### 2. Install Dependencies (Recommended)
```bash
# Install all dependencies and test online model loading
python install_dependencies.py
```

### 3. Manual Installation (Alternative)
```bash
# Create conda environment
conda create -n semisam python=3.10 -y
conda activate semisam

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
conda install numpy scipy matplotlib scikit-learn tqdm pillow pandas -y
conda install -c conda-forge simpleitk nibabel torchio opencv huggingface_hub -y
conda install -c huggingface transformers -y

# Install MedSAM2 package
pip install git+https://github.com/bowang-lab/MedSAM2.git
```

### 4. Verify Installation
```bash
python test_model_loading.py
```

## 🏗️ Architecture

### Student Model: SlicingSAM3D
```
Input: [B, 1, D, H, W] 3D volume
├── Slice along D dimension → [B, 1, H, W] 2D slices
├── MedSAM ViT Encoder → [B, 256, H//16, W//16] features per slice
├── Stack features → [B, 256, D, H//16, W//16] 3D features
└── 3D Decoder (4 layers) → [B, 1, D, H, W] segmentation mask
```

### Teacher Models
- **MedSAM ViT**: 2D SAM model for slice-by-slice processing
- **MedSAM2**: 3D model for volume-level processing with prompts

## 📖 Usage

### Basic Training
```python
from config import Config
from model import get_student_model, get_teacher_model
from train import train_semi

# Load configuration
config = Config()
config.data_path = 'path/to/your/data'
config.batch_size = 4

# Load models (automatically from HuggingFace Hub)
student = get_student_model(config)
teacher = get_teacher_model(config, 'medsam2')

# Train
train_semi(config, student, teacher, optimizer, train_loader, unlabeled_loader, device)
```

### Model Loading
```python
from model import get_medsam_vit_model, get_medsam2_model

# Load MedSAM ViT (from HuggingFace Hub)
processor, model = get_medsam_vit_model(device='cuda')

# Load MedSAM2 (from HuggingFace Hub)
model = get_medsam2_model(device='cuda')
```

### Direct HuggingFace Loading
```python
# Load model directly from HuggingFace Hub
from transformers import AutoProcessor, AutoModelForMaskGeneration

processor = AutoProcessor.from_pretrained("wanglab/medsam-vit-base")
model = AutoModelForMaskGeneration.from_pretrained("wanglab/medsam-vit-base")
```

## ⚙️ Configuration

### Model Selection
```python
# In config.py
student_model = 'slicing-sam3d'  # Custom 3D model
teacher_models = 'medsam2'       # or 'medsam-vit'

# Model paths (fallback only)
medsam_vit_path = './yiming_models_hgf/medsam-vit-base'
medsam2_path = './yiming_models_hgf/MedSAM2'
```

### Training Phases
1. **Warmup**: Train student model on labeled data only
2. **Semi-supervised**: Use teacher models for pseudo-label generation
3. **Fine-tune**: Final training on all data

## 📁 Project Structure

```
semi-sam-plus/
├── model.py                 # Model definitions and loading
├── config.py                # Configuration settings
├── train.py                 # Training loops
├── utils.py                 # Utility functions
├── data.py                  # Data loading utilities
├── install_dependencies.py  # Dependency installation script
├── install_models.py        # Local model installation (fallback)
├── install_models.sh        # Bash installation script
├── test_model_loading.py    # Setup verification
├── PIPELINE_SETUP.md        # Detailed setup guide
├── DOWNLOAD_MODELS.md       # Model download documentation
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Troubleshooting

### Common Issues

1. **Internet Connection Issues**
   ```
   ConnectionError: Failed to download model
   ```
   **Solution**: Check internet connection or use local model installation

2. **Missing Dependencies**
   ```
   ImportError: No module named 'medsam2'
   ```
   **Solution**: `pip install git+https://github.com/bowang-lab/MedSAM2.git`

3. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use CPU for testing

4. **HuggingFace Hub API Issues**
   ```
   TypeError: unexpected keyword argument 'local_dir'
   ```
   **Solution**: Update huggingface_hub: `pip install --upgrade huggingface_hub`

### Testing
Run the test script to diagnose issues:
```bash
python test_model_loading.py
```

### Fallback to Local Models
If online loading fails, you can still use local models:
```bash
# Install models locally
python install_models.py

# The pipeline will automatically fallback to local models
```

## 📚 Documentation

- [Pipeline Setup Guide](PIPELINE_SETUP.md) - Detailed setup instructions
- [Model Download Guide](DOWNLOAD_MODELS.md) - Local model installation guide
- [Student Model Building](student_model_building.md) - Student model details
- [Teacher Model Building](teacher_model_buidling.md) - Teacher model details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MedSAM2](https://github.com/bowang-lab/MedSAM2) - 3D medical image segmentation model
- [MedSAM ViT](https://huggingface.co/wanglab/medsam-vit-base) - 2D medical image segmentation model
- [SAM2](https://github.com/facebookresearch/sam2) - Segment Anything Model 2

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the documentation files
3. Open an issue on GitHub with detailed error information

---

**Note**: The pipeline now prioritizes online model loading from HuggingFace Hub for easier deployment. Local model installation is available as a fallback option.
