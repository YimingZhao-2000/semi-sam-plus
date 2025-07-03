class Config:
    """Configuration for Semi-SAM+ 3D Medical Image Segmentation Pipeline."""
    # Data parameters
    # Input 3D volume: [B, 1, D, H, W]
    data_path = 'path/to/data'  # TODO: Update with actual data path
    batch_size = 4
    num_workers = 8
    voxel_spacing = (0.5, 0.5, 0.5)  # mm
    crop_size = (128, 128, 128)  # [D, H, W] for input volume
    label_format = 'one-hot'  # or 'binary'
    labeled_ratio = 0.5  # |D_L|:|D_U| ≈ 1:1

    # Model paths (local yiming_models_hgf directory)
    models_dir = './yiming_models_hgf'
    medsam_vit_path = './yiming_models_hgf/medsam-vit-base'
    medsam2_path = './yiming_models_hgf/MedSAM2'
    
    # Model selection
    student_model = 'slicing-sam3d'  # Custom Slicing SAM3D model (uses MedSAM ViT as frozen encoder)
    teacher_models = 'medsam2'  # List of teacher models to use ('medsam2' or 'medsam-vit')
    model_params = {}

    # Training phases
    phase = 'warmup'  # Options: warmup, semi, finetune
    warmup_epochs = 50
    semi_epochs = 120
    finetune_epochs = 50
    lr_warmup = 2e-3
    lr_semi = 1e-3
    lr_finetune = 5e-4
    poly_lr = True

    # Loss weights and schedules
    lambda_max = 1.0
    beta_max = 0.3
    # Ramp schedules now handled by GPU tensor logic in train_semi
    
    # EMA
    ema_decay = 0.99

    # Checkpointing/logging
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    export_format = 'nii.gz'  # or 'dcmseg'

    # Hardware
    device = 'cuda'
    use_amp = True  # mixed precision
    grad_accum_steps = 1

    # Risk mitigation
    patch_sliding = True
    dataloader_cache = True
    prompt_var_threshold = 0.1

    # Pseudo label mask prompt size for MedSAM2 (downsampled mask input)
    pseudo_label_prompt_size = (32, 32, 32)

    # Slicing SAM3D specific settings
    # Slicing: input [B, 1, D, H, W] -> slices [B, 1, H, W] (for each d in D)
    # Encoder: each slice [B, 3, H, W] -> feature [B, 256, H//16, W//16]
    # Stack: [B, 256, D, H//16, W//16] (F)
    # Decoder: 4 layers, upsample x2 at each except last, output [B, 1, D, H, W]
    slicing_sam3d_settings = {
        'encoder_frozen': True,  # Freeze MedSAM ViT encoder
        'slice_step': 1,  # Step size for slicing (1 = every slice)
        'encoder_channels': 256,  # ViT-B output channels
        'decoder_channels': [256, 128, 64, 32],  # 3D decoder channel progression
        'decoder_levels': 4,  # Number of decoder levels (like U-Net)
        'use_skip_connections': True,  # U-Net style skip connections (not used in current model)
        'output_channels': 1,  # Binary segmentation output
        'activation': 'sigmoid'  # Output activation
    }
    
    # Model-specific settings
    medsam_vit_settings = {
        'use_processor': True,
        'max_length': 512,
        'num_beams': 1,
        'frozen': True  # Keep encoder frozen
    }
    
    medsam2_settings = {
        'checkpoint_name': 'MedSAM2_latest.pt',
        'use_memory_attention': True,
        'memory_frames': 10
    } 