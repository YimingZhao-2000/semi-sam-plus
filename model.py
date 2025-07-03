from transformers import AutoProcessor, AutoModelForMaskGeneration
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import subprocess
import numpy as np

# =====================
# Residual Dual Conv3D Block (Refined)
# =====================
class ResidualDualConv3D(nn.Module):
    """
    Refined Residual Dual Conv3D Block:
    r1 = IN(W2 * σ(IN(W1 * g)))
    r2 = IN(W0 * g)
    g' = U2(σ(r1 + r2))
    W0/W1/W2: 3x3x3 Conv, IN: InstanceNorm3d, σ: LeakyReLU, U2: trilinear upsample x2
    """
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        # W1, W2, W0
        self.conv_w1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.in1 = nn.InstanceNorm3d(out_channels)
        self.conv_w2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels)
        self.conv_w0 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.in0 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
    def forward(self, g):
        r1 = self.in1(self.conv_w1(g))
        r1 = self.act(r1)
        r1 = self.in2(self.conv_w2(r1))
        r2 = self.in0(self.conv_w0(g))
        out = self.act(r1 + r2)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
        return out

# =====================
# Dice + Cross-Entropy Loss (Refined)
# =====================
class DiceCELoss(nn.Module):
    """
    L = CE + (1 - SoftDice)
    SoftDice = 2 <Y, Y_hat> / (||Y||^2 + ||Y_hat||^2)
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss()  # For binary segmentation
    def forward(self, pred, target):
        ce = self.ce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        num = 2 * (pred_sigmoid * target).sum()
        denom = (pred_sigmoid**2).sum() + (target**2).sum() + 1e-6
        dice = num / denom
        return ce + (1 - dice)

class MultiScaleDeepSupervision(nn.Module):
    """
    Multi-scale deep supervision with weights (1/15)*(8,4,2,1)
    """
    def __init__(self):
        super().__init__()
        self.weights = [8/15, 4/15, 2/15, 1/15]
        self.loss_fn = DiceCELoss()
    def forward(self, preds, target):
        total = 0.0
        for i, pred in enumerate(preds):
            tgt = F.interpolate(target, size=pred.shape[2:], mode='nearest')
            total += self.weights[i] * self.loss_fn(pred, tgt)
        return total

# =====================
# Slicing SAM3D Model (Refined Decoder)
# =====================
class SlicingSAM3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.slice_step = config.slicing_sam3d_settings['slice_step']
        self.encoder_channels = config.slicing_sam3d_settings['encoder_channels']
        self.decoder_channels = config.slicing_sam3d_settings['decoder_channels']
        self.decoder_levels = config.slicing_sam3d_settings['decoder_levels']
        self.output_channels = config.slicing_sam3d_settings['output_channels']
        self.encoder_processor, self.encoder = self._load_frozen_encoder()
        # Decoder: 4 layers, each upsample except last
        self.decoder = nn.ModuleList([
            ResidualDualConv3D(self.encoder_channels, self.decoder_channels[0], upsample=True),
            ResidualDualConv3D(self.decoder_channels[0], self.decoder_channels[1], upsample=True),
            ResidualDualConv3D(self.decoder_channels[1], self.decoder_channels[2], upsample=True),
            ResidualDualConv3D(self.decoder_channels[2], self.decoder_channels[3], upsample=False),
        ])
        self.output_layers = nn.ModuleList([
            nn.Conv3d(self.decoder_channels[i], self.output_channels, 1)
            for i in range(self.decoder_levels)
        ])
        self.loss_fn = MultiScaleDeepSupervision()
    def _load_frozen_encoder(self):
        processor, model = get_medsam_vit_model(self.config.device)
        if self.config.slicing_sam3d_settings['encoder_frozen']:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        return processor, model
    def _slice_and_encode(self, volume):
        B, C, D, H, W = volume.shape
        features_list = []
        for d in range(0, D, self.slice_step):
            slice_2d = volume[:, :, d, :, :]
            slice_rgb = slice_2d.repeat(1, 3, 1, 1)
            slice_rgb = (slice_rgb * 255).clamp(0, 255).to(torch.uint8)
            with torch.no_grad():
                encoded_features = self._extract_encoder_features(slice_rgb)
                features_list.append(encoded_features)
        stacked_features = torch.stack(features_list, dim=2)
        return stacked_features
    def _extract_encoder_features(self, slice_rgb):
        """
        Extract features from MedSAM ViT encoder.
        Args:
            slice_rgb: [B, 3, H, W] RGB tensor in uint8 format
        Returns:
            features: [B, encoder_channels, H//16, W//16] feature tensor
        """
        # Convert to PIL images for SAM processor
        B, C, H, W = slice_rgb.shape
        features_list = []
        
        for b in range(B):
            # Convert tensor to PIL image
            img = slice_rgb[b].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            img = img.astype(np.uint8)
            
            # Process with SAM encoder
            with torch.no_grad():
                # Use the processor to get the right format
                inputs = self.encoder_processor(img, return_tensors="pt").to(slice_rgb.device)
                
                # Extract features from the vision encoder
                vision_outputs = self.encoder.vision_encoder(**inputs)
                image_embeddings = vision_outputs.last_hidden_state  # [1, 1024, H//16, W//16]
                
                # Reshape to expected format
                features = image_embeddings.squeeze(0)  # [1024, H//16, W//16]
                features_list.append(features)
        
        # Stack batch features
        features = torch.stack(features_list, dim=0)  # [B, 1024, H//16, W//16]
        
        # Project to expected encoder channels if needed
        if features.shape[1] != self.encoder_channels:
            # Add a projection layer if SAM features don't match expected channels
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Conv2d(
                    features.shape[1], self.encoder_channels, 1
                ).to(features.device)
            features = self.feature_projection(features)
        
        return features
    def _decode_3d(self, features):
        x = features
        outputs = []
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            outputs.append(self.output_layers[i](x))
        return outputs
    def forward(self, volume, target=None):
        features = self._slice_and_encode(volume)
        multi_scale_outputs = self._decode_3d(features)
        final_output = multi_scale_outputs[-1]
        if self.config.slicing_sam3d_settings['activation'] == 'sigmoid':
            final_output = torch.sigmoid(final_output)
        elif self.config.slicing_sam3d_settings['activation'] == 'softmax':
            final_output = F.softmax(final_output, dim=1)
        loss = None
        if target is not None:
            loss = self.loss_fn(multi_scale_outputs, target)
            return final_output, loss
        return final_output

# =====================
# Git LFS Helper Functions
# =====================
def ensure_git_lfs():
    """Ensure git-lfs is installed and initialized."""
    try:
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("git-lfs is not installed. Please install it from https://git-lfs.com")

def clone_repo_with_lfs(repo_url, local_path, skip_lfs=False):
    """
    Clone repository with git LFS support.
    Args:
        repo_url: Repository URL (e.g., 'https://huggingface.co/wanglab/medsam-vit-base')
        local_path: Local path to clone to
        skip_lfs: If True, skip downloading large files (just pointers)
    """
    ensure_git_lfs()
    
    if os.path.exists(local_path):
        print(f"Repository already exists at {local_path}")
        return local_path
    
    if skip_lfs:
        env = os.environ.copy()
        env['GIT_LFS_SKIP_SMUDGE'] = '1'
        subprocess.run(['git', 'clone', repo_url, local_path], env=env, check=True)
    else:
        subprocess.run(['git', 'clone', repo_url, local_path], check=True)
    
    return local_path

# =====================
# MedSAM ViT Loader
# =====================
def get_medsam_vit_model(device='cuda', repo_path=None, skip_lfs=False):
    """
    Load MedSAM ViT model and processor from HuggingFace Hub.
    Falls back to local path if online loading fails.
    Args:
        device: Device to load model on
        repo_path: Local path to model (if None, uses default yiming_models_hgf path)
        skip_lfs: If True, skip downloading large files initially (for cloning)
    """
    print("Loading MedSAM ViT from HuggingFace Hub...")
    
    try:
        # Try to load directly from HuggingFace Hub
        processor = AutoProcessor.from_pretrained("wanglab/medsam-vit-base")
        model = AutoModelForMaskGeneration.from_pretrained("wanglab/medsam-vit-base").to(device)
        print("✓ MedSAM ViT loaded successfully from HuggingFace Hub")
        return processor, model
        
    except Exception as e:
        print(f"⚠️  Online loading failed: {e}")
        print("Falling back to local model...")
        
        # Fallback to local path
        if repo_path is None:
            repo_path = './yiming_models_hgf/medsam-vit-base'
        
        # Check if local path exists
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"MedSAM ViT model not found at {repo_path}. "
                                   "Please run the installation script first: python install_models.py")
        
        # Verify model files exist (check for huggingface-cli downloaded structure)
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        missing_files = []
        for file in required_files:
            file_path = os.path.join(repo_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"Missing files: {missing_files}")
            print("This might be due to incomplete download. Please run:")
            print("python install_models.py")
            raise FileNotFoundError(f"Missing MedSAM ViT model files: {missing_files}. "
                                   "Please run the installation script to download the complete model.")
        
        print(f"Loading MedSAM ViT from local path: {repo_path}")
        
        # Load model and processor from local path
        try:
            processor = AutoProcessor.from_pretrained(repo_path)
            model = AutoModelForMaskGeneration.from_pretrained(repo_path).to(device)
            print("✓ MedSAM ViT loaded successfully from local path")
            return processor, model
        except Exception as e:
            print(f"Error loading from local path: {e}")
            print("The model files might be corrupted. Please reinstall:")
            print("python install_models.py")
            raise

def medsam_vit_inference(processor, model, image):
    """
    Run mask generation on a given image using MedSAM ViT (automatic mask generation).
    Args:
        image: np.ndarray or PIL.Image
    Returns:
        mask: np.ndarray
    """
    inputs = processor(image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    # Post-process as needed
    return outputs.masks.cpu().numpy()

# =====================
# MedSAM2 Loader
# =====================
def get_medsam2_model(device='cuda', repo_path=None, skip_lfs=False):
    """
    Load MedSAM2 model from HuggingFace Hub or local path.
    Args:
        device: Device to load model on
        repo_path: Local path to model (if None, uses default yiming_models_hgf path)
        skip_lfs: If True, skip downloading large files initially (for cloning)
    """
    print("Loading MedSAM2 from HuggingFace Hub...")
    
    try:
        # Try to load directly from HuggingFace Hub
        from huggingface_hub import hf_hub_download
        import tempfile
        
        # Download the latest checkpoint to a temporary location
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = hf_hub_download(
                repo_id="wanglab/MedSAM2",
                filename="MedSAM2_latest.pt",
                cache_dir=temp_dir
            )
            
            # Try to import MedSAM2 from installed package
            try:
                from medsam2 import MedSAM2
                print("MedSAM2 package found, loading model...")
                
                # Initialize model
                model = MedSAM2()
                
                # Load checkpoint
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint)
                model.to(device)
                model.eval()
                
                print("✓ MedSAM2 loaded successfully from HuggingFace Hub")
                return model
                
            except ImportError:
                print("MedSAM2 package not installed. Please install it first:")
                print("pip install git+https://github.com/bowang-lab/MedSAM2.git")
                raise ImportError("MedSAM2 package is required. Please install it using the command above.")
                
    except Exception as e:
        print(f"⚠️  Online loading failed: {e}")
        print("Falling back to local model...")
        
        # Fallback to local path
        if repo_path is None:
            repo_path = './yiming_models_hgf/MedSAM2'
        
        # Check if local path exists
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"MedSAM2 model not found at {repo_path}. "
                                   "Please run the installation script first: python install_models.py")
        
        # Path to the latest checkpoint
        ckpt_path = os.path.join(repo_path, 'MedSAM2_latest.pt')
        
        if not os.path.exists(ckpt_path):
            print(f"MedSAM2 checkpoint not found at {ckpt_path}")
            print("Available files in MedSAM2 directory:")
            if os.path.exists(repo_path):
                for file in os.listdir(repo_path):
                    file_path = os.path.join(repo_path, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"  - {file} ({size:,} bytes)")
            
            raise FileNotFoundError(f"MedSAM2 checkpoint not found at {ckpt_path}. "
                                   "Please run the installation script to download the complete model.")
        
        print(f"Loading MedSAM2 from local path: {ckpt_path}")
        
        # Try to import MedSAM2 from installed package
        try:
            from medsam2 import MedSAM2
            print("MedSAM2 package found, loading model...")
            
            # Initialize model
            model = MedSAM2()
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            
            print("✓ MedSAM2 loaded successfully from local path")
            return model
            
        except ImportError:
            print("MedSAM2 package not installed. Please install it first:")
            print("pip install git+https://github.com/bowang-lab/MedSAM2.git")
            raise ImportError("MedSAM2 package is required. Please install it using the command above.")
        
        except Exception as e:
            print(f"Error loading MedSAM2 model: {e}")
            print("Please ensure MedSAM2 is properly installed and the checkpoint is valid.")
            raise

# =====================
# Model Definitions
# =====================
def get_student_model(config):
    """Return the Student model (e.g., SlicingSAM3D, MedSAM ViT)."""
    if config.student_model.lower() in ['slicing-sam3d', 'slicing_sam3d']:
        return SlicingSAM3D(config)
    elif config.student_model.lower() in ['medsam-vit', 'medsamvit', 'wanglab/medsam-vit-base']:
        processor, model = get_medsam_vit_model(config.device)
        return model  # Return just the model for student
    # TODO: Implement other student models
    raise NotImplementedError('Implement other student models.')

def get_teacher_model(config, teacher_type):
    """Return the Teacher model (TinySAM-Med3D, SAM-Med3D, MedSAM2, MedSAM ViT)."""
    if teacher_type.lower() in ['medsam-vit', 'medsamvit', 'wanglab/medsam-vit-base']:
        processor, model = get_medsam_vit_model(config.device)
        return model
    elif teacher_type.lower() in ['medsam2', 'medsam-2']:
        return get_medsam2_model(config.device)
    # TODO: Implement other teacher models
    raise NotImplementedError(f'Implement {teacher_type} model.')

# =====================
# EMA Update
# =====================
def update_ema(student_model, teacher_model, decay):
    # EMA update: theta'_t = decay * theta'_{t-1} + (1-decay) * theta_t
    with torch.no_grad():
        for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay) 