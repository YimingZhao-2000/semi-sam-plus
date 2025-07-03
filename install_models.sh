#!/bin/bash

# MedSAM2 and MedSAM ViT Installation Script
# Updated to use huggingface-cli for reliable model downloading

set -e  # Exit on any error

echo "MedSAM2 and MedSAM ViT Installation Script"
echo "Using huggingface-cli for reliable downloading"
echo "=================================================="

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "✗ huggingface-cli is not installed"
    echo "Installing huggingface-cli..."
    pip install huggingface_hub
    echo "✓ huggingface-cli installed successfully"
else
    echo "✓ huggingface-cli is installed"
fi

# Create base directory
mkdir -p yiming_models_hgf

# Download MedSAM ViT Base
echo ""
echo "Downloading MedSAM ViT Base..."
MEDSAM_VIT_DIR="yiming_models_hgf/medsam-vit-base"
mkdir -p "$MEDSAM_VIT_DIR"

if huggingface-cli download wanglab/medsam-vit-base --local-dir "$MEDSAM_VIT_DIR" --local-dir-use-symlinks False; then
    echo "✓ MedSAM ViT downloaded to: $MEDSAM_VIT_DIR"
else
    echo "✗ Failed to download MedSAM ViT"
    exit 1
fi

# Download MedSAM2
echo ""
echo "Downloading MedSAM2..."
MEDSAM2_DIR="yiming_models_hgf/MedSAM2"
mkdir -p "$MEDSAM2_DIR"

# List of MedSAM2 checkpoint files to download
checkpoint_files=(
    "MedSAM2_latest.pt"
    "MedSAM2_2411.pt"
    "MedSAM2_US_Heart.pt"
    "MedSAM2_MRI_LiverLesion.pt"
    "MedSAM2_CTLesion.pt"
)

# Additional files
additional_files=(
    "README.md"
    "config.json"
)

downloaded_count=0

# Download checkpoint files
for filename in "${checkpoint_files[@]}"; do
    echo "  Downloading $filename..."
    if huggingface-cli download wanglab/MedSAM2 "$filename" --local-dir "$MEDSAM2_DIR" --local-dir-use-symlinks False; then
        echo "    ✓ $filename downloaded"
        ((downloaded_count++))
    else
        echo "    ✗ Failed to download $filename"
    fi
done

# Download additional files
for filename in "${additional_files[@]}"; do
    echo "  Downloading $filename..."
    if huggingface-cli download wanglab/MedSAM2 "$filename" --local-dir "$MEDSAM2_DIR" --local-dir-use-symlinks False; then
        echo "    ✓ $filename downloaded"
        ((downloaded_count++))
    else
        echo "    ✗ Failed to download $filename"
    fi
done

if [ $downloaded_count -gt 0 ]; then
    echo "✓ MedSAM2 downloaded $downloaded_count files to: $MEDSAM2_DIR"
else
    echo "✗ No MedSAM2 files were downloaded"
    exit 1
fi

# Verify files
echo ""
echo "Verifying downloaded files..."

# Verify MedSAM ViT files
echo "MedSAM ViT files:"
if [ -f "$MEDSAM_VIT_DIR/config.json" ]; then
    size=$(stat -f%z "$MEDSAM_VIT_DIR/config.json" 2>/dev/null || stat -c%s "$MEDSAM_VIT_DIR/config.json" 2>/dev/null || echo "unknown")
    echo "  ✓ config.json ($size bytes)"
else
    echo "  ✗ config.json missing"
fi

if [ -f "$MEDSAM_VIT_DIR/pytorch_model.bin" ]; then
    size=$(stat -f%z "$MEDSAM_VIT_DIR/pytorch_model.bin" 2>/dev/null || stat -c%s "$MEDSAM_VIT_DIR/pytorch_model.bin" 2>/dev/null || echo "unknown")
    echo "  ✓ pytorch_model.bin ($size bytes)"
else
    echo "  ✗ pytorch_model.bin missing"
fi

# Verify MedSAM2 files
echo "MedSAM2 files:"
if [ -f "$MEDSAM2_DIR/MedSAM2_latest.pt" ]; then
    size=$(stat -f%z "$MEDSAM2_DIR/MedSAM2_latest.pt" 2>/dev/null || stat -c%s "$MEDSAM2_DIR/MedSAM2_latest.pt" 2>/dev/null || echo "unknown")
    echo "  ✓ MedSAM2_latest.pt ($size bytes)"
else
    echo "  ✗ MedSAM2_latest.pt missing"
fi

if [ -f "$MEDSAM2_DIR/README.md" ]; then
    size=$(stat -f%z "$MEDSAM2_DIR/README.md" 2>/dev/null || stat -c%s "$MEDSAM2_DIR/README.md" 2>/dev/null || echo "unknown")
    echo "  ✓ README.md ($size bytes)"
else
    echo "  ✗ README.md missing"
fi

# Summary
echo ""
echo "=================================================="
echo "INSTALLATION SUMMARY"
echo "=================================================="
echo "Base directory: $(pwd)/yiming_models_hgf"
echo "MedSAM ViT Base: $MEDSAM_VIT_DIR"
echo "MedSAM2: $MEDSAM2_DIR"
echo ""
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install transformers torch"
echo "2. Install MedSAM2 package: pip install git+https://github.com/bowang-lab/MedSAM2.git"
echo "3. Test setup: python test_model_loading.py" 