import os
import torch
import torch.nn.functional as F
import numpy as np
# Optionally use torchio for medical image augmentations
try:
    import torchio as tio
    HAS_TIO = True
except ImportError:
    HAS_TIO = False

# =====================
# Checkpointing
# =====================
def save_checkpoint(model, optimizer, epoch, path):
    """Save model and optimizer state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load model and optimizer state."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

# =====================
# Slicing & Reassembly
# =====================
def slice_volume(volume, step):
    """Slice 3D volume along z with step size."""
    # TODO: Implement slicing
    pass

def reassemble_mask(slices, positions, target_shape):
    """Reassemble mask from slices."""
    # TODO: Implement reassembly
    pass

# =====================
# Augmentation (with transform/inverse)
# =====================
def strong_aug(volume):
    # Example: random affine, flip, noise
    if HAS_TIO:
        t = tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.75),
            tio.RandomFlip(axes=(0, 1, 2), p=0.5),
            tio.RandomNoise(p=0.5)
        ])
        aug = t(volume)
        return aug, t
    else:
        # Stub: identity transform
        return volume, lambda x: x

def weak_aug(volume):
    # Example: light affine, flip
    if HAS_TIO:
        t = tio.Compose([
            tio.RandomAffine(scales=(0.98, 1.02), degrees=3, translation=2, p=0.5),
            tio.RandomFlip(axes=(0, 1, 2), p=0.2)
        ])
        aug = t(volume)
        return aug, t
    else:
        # Stub: identity transform
        return volume, lambda x: x

# =====================
# Prompt Queue Generation (for teacher)
# =====================
def generate_prompt_queue(coarse_mask, N=4, mode='centroid_or_random', mask_prompt=True, target_size=(32,32,32)):
    """
    Generate N diverse prompts for each sample, including:
      - point prompts (centroid or random point)
      - mask prompt (low-res mask, always downsampled to target_size)
    Returns a list of dicts: [{"point_coords": [[z, y, x]], "point_labels": [1], "mask_input": mask}, ...]
    """
    B = coarse_mask.shape[0]
    prompt_queue = []
    def get_point_prompt(mask):
        idx = (mask > 0.5).nonzero(as_tuple=False)
        if idx.numel() > 0:
            if mode == 'centroid_or_random':
                centroid = idx.float().mean(dim=0)
                if torch.rand(1).item() < 0.5:
                    pt = centroid.round().long().tolist()
                else:
                    rand_idx = idx[torch.randint(0, idx.shape[0], (1,))]
                    pt = rand_idx.tolist()[0]
            else:
                rand_idx = idx[torch.randint(0, idx.shape[0], (1,))]
                pt = rand_idx.tolist()[0]
            return pt
        else:
            return [0, 0, 0]
    for _ in range(N):
        batch_prompts = []
        for b in range(B):
            mask = coarse_mask[b, 0]  # [D, H, W]
            pt = get_point_prompt(mask)
            prompt = {"point_coords": [pt], "point_labels": [1]}
            if mask_prompt:
                mask_input = mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                mask_input = F.interpolate(mask_input, size=target_size, mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
                prompt["mask_input"] = mask_input
            batch_prompts.append(prompt)
        prompt_queue.append(batch_prompts)
    return prompt_queue  # List of [B, dict]

# =====================
# Pseudo-label Generation & Filtering
# =====================
def map_mask_to_original_space(mask, t_strong):
    # Map mask from augmented (student) space back to original space using inverse transform
    if hasattr(t_strong, 'inverse'):
        return t_strong.inverse()(mask)
    elif hasattr(t_strong, '__call__') and hasattr(t_strong, 'is_identity') and not t_strong.is_identity:
        # If t_strong is a callable with an 'inverse' method
        return t_strong.inverse()(mask)
    else:
        return mask  # Identity if no transform

def generate_pseudo_labels(teacher, x_u, coarse_mask, t_strong=None, N=4, u_th=0.1, target_size=(32,32,32)):
    """
    Generate pseudo-labels using both point and mask prompts.
    Supports both MedSAM ViT (2D SAM) and MedSAM2 (3D) models.
    """
    # Detect model type based on teacher model class name
    model_class_name = teacher.__class__.__name__
    
    if 'AutoModelForMaskGeneration' in str(type(teacher)) or 'SamModel' in model_class_name:
        # MedSAM ViT: 2D SAM model, process slice by slice
        return _generate_pseudo_labels_medsam_vit(teacher, x_u, coarse_mask, t_strong, N, u_th)
    elif 'MedSAM2' in model_class_name:
        # MedSAM2: 3D model, use 3D API
        return _generate_pseudo_labels_medsam2(teacher, x_u, coarse_mask, t_strong, N, u_th, target_size)
    else:
        raise ValueError(f"Unknown teacher model type: {model_class_name}. Expected MedSAM ViT or MedSAM2.")

def _generate_pseudo_labels_medsam_vit(teacher, x_u, coarse_mask, t_strong=None, N=4, u_th=0.1):
    """
    Generate pseudo-labels using MedSAM ViT (2D SAM model).
    Process each slice of the 3D volume separately without prompts.
    """
    B, C, D, H, W = x_u.shape
    all_preds = []
    
    for d in range(D):
        # Extract 2D slice
        slice_2d = x_u[:, :, d, :, :]  # [B, C, H, W]
        
        slice_preds = []
        for b in range(B):
            # Convert to PIL or numpy for SAM processor
            img = slice_2d[b].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            if img.shape[2] == 1:
                img = img.repeat(3, axis=2)  # Convert to RGB
            
            with torch.no_grad():
                # Use SAM's generate method without prompts (automatic mask generation)
                # Convert to PIL Image for SAM
                from PIL import Image
                img_pil = Image.fromarray(img.astype(np.uint8))
                
                # Generate masks using SAM
                outputs = teacher.generate(
                    pixel_values=slice_2d[b:b+1].to(teacher.device),
                    return_dict=True
                )
                pred = outputs.masks[0]  # [1, H, W]
                slice_preds.append(pred)
        
        # Stack predictions for this slice
        slice_preds = torch.stack(slice_preds, dim=0)  # [B, 1, H, W]
        all_preds.append(slice_preds)
    
    # Stack all slices back to 3D
    preds = torch.stack(all_preds, dim=2)  # [B, 1, D, H, W]
    
    # For MedSAM ViT, we only have one prediction per slice, so no variance computation
    # Just use the prediction directly as pseudo label
    pseudo_label = preds
    mask_conf = torch.ones_like(pseudo_label)  # All confident for MedSAM ViT
    
    # Map back to original space if needed
    if t_strong is not None:
        pseudo_label = map_mask_to_original_space(pseudo_label, t_strong)
        mask_conf = map_mask_to_original_space(mask_conf, t_strong)
    
    return pseudo_label, mask_conf

def _generate_pseudo_labels_medsam2(teacher, x_u, coarse_mask, t_strong=None, N=4, u_th=0.1, target_size=(32,32,32)):
    """
    Generate pseudo-labels using MedSAM2 (3D model).
    Use the 3D API with point_coords, point_labels, mask_input.
    """
    prompt_queue = generate_prompt_queue(coarse_mask, N, mask_prompt=True, target_size=target_size)
    B = coarse_mask.shape[0]
    # Flatten prompts for batch processing
    flat_point_coords = []
    flat_point_labels = []
    flat_mask_inputs = []
    for prompts in prompt_queue:
        for p in prompts:
            flat_point_coords.append(p["point_coords"][0])
            flat_point_labels.append(p["point_labels"][0])
            flat_mask_inputs.append(p["mask_input"])
    # Convert to tensors
    points_tensor = torch.tensor(flat_point_coords, device=x_u.device, dtype=torch.float32)  # [N*B, 3]
    labels_tensor = torch.tensor(flat_point_labels, device=x_u.device, dtype=torch.int64)    # [N*B]
    masks_tensor = torch.stack(flat_mask_inputs).to(x_u.device)                              # [N*B, D',H',W']
    # Normalize point coordinates to [0,1]
    D, H, W = x_u.shape[-3:]
    points_tensor[:, 0] /= D
    points_tensor[:, 1] /= H
    points_tensor[:, 2] /= W
    # Add batch and K dims for MedSAM2 API ([B*N, K, 3], [B*N, K])
    points_tensor = points_tensor.unsqueeze(1)  # [N*B, 1, 3]
    labels_tensor = labels_tensor.unsqueeze(1)  # [N*B, 1]
    masks_tensor = masks_tensor.unsqueeze(1)    # [N*B, 1, D',H',W']
    # Repeat x_u for each prompt
    x_u_rep = x_u.repeat_interleave(N, dim=0)  # [N*B, ...]
    with torch.no_grad():
        # Call MedSAM2 with proper API
        try:
            # Try the expected MedSAM2 API
            preds_bc, _ = teacher(
                x_u_rep,
                point_coords=points_tensor,
                point_labels=labels_tensor,
                mask_input=masks_tensor,
                multimask_output=False
            )  # preds_bc: [N*B, 1, D, H, W] or similar
        except Exception as e:
            print(f"MedSAM2 API call failed: {e}")
            print("Trying alternative API format...")
            # Try alternative API format if the first one fails
            preds_bc = teacher(
                x_u_rep,
                point_coords=points_tensor,
                point_labels=labels_tensor,
                mask_input=masks_tensor
            )
    
    # Reshape back to [N, B, ...]
    preds = preds_bc.view(N, B, *preds_bc.shape[1:])
    mean_pred = preds.mean(dim=0)
    var_pred = preds.var(dim=0)
    
    # Compute confidence based on variance
    mask_conf = torch.exp(-var_pred / u_th)
    mask_conf = torch.clamp(mask_conf, 0, 1)
    
    # Apply confidence threshold
    pseudo_label = mean_pred * (mask_conf > u_th).float()
    
    # Map back to original space if needed
    if t_strong is not None:
        pseudo_label = map_mask_to_original_space(pseudo_label, t_strong)
        mask_conf = map_mask_to_original_space(mask_conf, t_strong)
    
    return pseudo_label, mask_conf

def variance_filter(pseudo_labels, threshold):
    var = torch.var(pseudo_labels, dim=0)
    mask_conf = (var < threshold).float()
    return mask_conf

# =====================
# Export
# =====================
def export_mask(mask, path, format='nii.gz'):
    # TODO: Implement export to nii.gz or dcmseg
    pass

def region_consistency_loss(student_pred, pseudo_label, mask_conf):
    """
    Compute region consistency loss between student predictions and pseudo labels.
    L_rs-con = Σ M_conf·||P_s − P_f||² / Σ M_conf
    
    Args:
        student_pred: Student model prediction [B, 1, D, H, W]
        pseudo_label: Pseudo label from teacher [B, 1, D, H, W]
        mask_conf: Confidence mask [B, 1, D, H, W]
    Returns:
        loss: Region consistency loss scalar
    """
    # Ensure inputs are on the same device
    device = student_pred.device
    pseudo_label = pseudo_label.to(device)
    mask_conf = mask_conf.to(device)
    
    # Compute squared difference
    diff = (student_pred - pseudo_label) ** 2
    
    # Weight by confidence mask
    masked_diff = diff * mask_conf
    
    # Compute weighted mean
    numerator = masked_diff.sum()
    denominator = mask_conf.sum() + 1e-6  # Avoid division by zero
    
    loss = numerator / denominator
    return loss

# Add more utility functions as needed 