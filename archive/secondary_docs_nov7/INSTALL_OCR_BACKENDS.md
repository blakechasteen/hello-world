# OCR Backend Installation Guide

**Date**: January 2025
**Status**: Complete guide for all OCR backends

---

## Overview

HoloLoom supports 3 OCR backends with automatic fallback:

1. **DeepSeek OCR** (EXCELLENT) - Best quality, requires CUDA
2. **Tesseract** (GOOD) - Good quality, CPU-only
3. **Fallback** (POOR) - Always works, filename extraction

---

## Option 1: Tesseract OCR (RECOMMENDED - CPU)

### Advantages
✅ Good quality (85-95% accuracy)
✅ CPU-only (no CUDA required)
✅ Fast (~200-300ms per image)
✅ Mature, well-tested
✅ Free and open source

### Installation

#### Windows

**Option A: Chocolatey (Recommended)**
```bash
choco install tesseract
```

**Option B: Manual Download**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer (tesseract-ocr-w64-setup-5.x.exe)
3. Add to PATH:
   ```
   C:\Program Files\Tesseract-OCR
   ```

**Option C: Scoop**
```bash
scoop install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

#### macOS
```bash
brew install tesseract
```

### Python Package

```bash
pip install pytesseract
```

### Verification

```bash
# Check Tesseract installation
tesseract --version

# Expected output:
# tesseract 5.x.x

# Test with Python
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

### Configuration

Create `.env` or set environment variable:

```bash
# Windows
set TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# Linux/Mac
export TESSERACT_PATH=/usr/bin/tesseract
```

Or in Python:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Performance

| Image Size | Time | Accuracy | CPU Usage |
|------------|------|----------|-----------|
| 1024x1024 | 200ms | 90-95% | 25-50% |
| 2048x2048 | 500ms | 92-97% | 50-75% |
| 4096x4096 | 1500ms | 93-98% | 75-100% |

---

## Option 2: DeepSeek OCR (BEST - CUDA Required)

### Advantages
✅ Excellent quality (95-99% accuracy)
✅ 10x compression ratio
✅ Structured markdown output
✅ Multiple resolution support
✅ Best for complex layouts

### Requirements

**Hardware**:
- NVIDIA GPU with CUDA 11.8+
- 8GB+ VRAM (16GB recommended)
- 16GB+ system RAM

**Software**:
- CUDA Toolkit 11.8+
- cuDNN 8.9+
- PyTorch 2.6.0+ with CUDA

### Installation

#### Step 1: Install CUDA Toolkit

**Windows**:
1. Download from: https://developer.nvidia.com/cuda-downloads
2. Run installer: `cuda_12.1.0_windows_network.exe`
3. Select components: CUDA Toolkit + Visual Studio Integration

**Linux (Ubuntu)**:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

#### Step 2: Install PyTorch with CUDA

```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install vLLM (Recommended)

```bash
pip install vllm==0.8.5
```

Or Transformers (Alternative):
```bash
pip install transformers>=4.40.0
```

#### Step 4: Install DeepSeek OCR

```bash
pip install deepseek-ocr
```

Or clone from source:
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
pip install -e .
```

### Verification

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Expected output:
# PyTorch: 2.6.0+cu121
# CUDA available: True
# CUDA version: 12.1
# Device: NVIDIA GeForce RTX 3080
```

### Usage

```python
from HoloLoom.spinningWheel import DeepSeekOCRSpinner

spinner = DeepSeekOCRSpinner(
    backend_type="vllm",  # or "transformers"
    resolution=1024,
    device="cuda"
)

result = await spinner.spin("receipt.jpg")
```

### Performance

| Image Size | Time (vLLM) | Time (Transformers) | Accuracy | VRAM |
|------------|-------------|---------------------|----------|------|
| 1024x1024 | 150ms | 300ms | 95-99% | 4GB |
| 2048x2048 | 300ms | 600ms | 96-99% | 6GB |
| 4096x4096 | 800ms | 1500ms | 97-99% | 12GB |

---

## Quick Start Guide

### For CPU Users (No CUDA)

Use **Tesseract** - works great, no GPU needed!

```bash
# Install Tesseract
choco install tesseract  # Windows
# OR
brew install tesseract   # macOS
# OR
sudo apt-get install tesseract-ocr  # Linux

# Install Python package
pip install pytesseract

# Test
python -c "import pytesseract; print('OK')"
```

### For GPU Users (CUDA Available)

Use **DeepSeek OCR** - best quality!

```bash
# 1. Install CUDA Toolkit (if not installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install vLLM
pip install vllm==0.8.5

# 4. Install DeepSeek OCR
pip install deepseek-ocr

# Test
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Backend Selection

HoloLoom automatically selects the best available backend:

```python
from HoloLoom.spinningWheel.ocr_backends import get_all_available_backends

# Automatic selection with fallback
backend = get_all_available_backends()
print(f"Using: {backend.get_name()} ({backend.get_quality().value})")

# Priority:
# 1. DeepSeek (if CUDA available) - EXCELLENT
# 2. Tesseract (if installed) - GOOD
# 3. Fallback (always works) - POOR
```

### Manual Selection

```python
from HoloLoom.spinningWheel.ocr_backends import (
    get_best_available_backend,
    TesseractOCRBackend,
    DeepSeekOCRBackend
)

# Force Tesseract
backend = TesseractOCRBackend()
if backend.is_available():
    result = await backend.extract_text("receipt.jpg")

# Force DeepSeek
from HoloLoom.spinningWheel.ocr_backends.deepseek import DeepSeekConfig
config = DeepSeekConfig(resolution=1024)
backend = DeepSeekOCRBackend(config)
if backend.is_available():
    result = await backend.extract_text("receipt.jpg")
```

---

## Troubleshooting

### Tesseract Issues

**Problem**: `tesseract is not recognized as an internal or external command`

**Solution**: Add to PATH
```bash
# Windows
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"

# Or in Python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Problem**: `TesseractNotFoundError`

**Solution**: Install Tesseract
```bash
choco install tesseract  # Windows
brew install tesseract   # macOS
```

### DeepSeek Issues

**Problem**: `CUDA out of memory`

**Solution**: Reduce resolution
```python
config = DeepSeekConfig(resolution=512)  # Try 512 or 640
```

**Problem**: `No CUDA GPUs are available`

**Solution**: Check CUDA installation
```bash
nvidia-smi  # Should show GPU info
nvcc --version  # Should show CUDA version
```

**Problem**: `ImportError: cannot import name 'LLM' from 'vllm'`

**Solution**: Install correct vLLM version
```bash
pip install vllm==0.8.5
```

### General Issues

**Problem**: `ModuleNotFoundError: No module named 'pytesseract'`

**Solution**:
```bash
pip install pytesseract
```

**Problem**: OCR results are poor

**Solutions**:
1. Increase image resolution
2. Improve image quality (brightness, contrast)
3. Try different backend (DeepSeek > Tesseract > Fallback)

---

## Recommendations

### For Development

**Tesseract** - Fast, good quality, works everywhere

```bash
choco install tesseract
pip install pytesseract
```

### For Production (CPU)

**Tesseract** - Reliable, battle-tested

```bash
# Linux production
sudo apt-get install tesseract-ocr
pip install pytesseract
```

### For Production (GPU)

**DeepSeek OCR** - Best quality, worth the setup

```bash
# Install CUDA + PyTorch + vLLM + DeepSeek
# See full guide above
```

### For Quick Testing

**Fallback** - Already works, no installation needed!

```python
from HoloLoom.spinningWheel import SchemaAwareReceiptSpinner

# Works out of the box (uses filename extraction)
spinner = SchemaAwareReceiptSpinner(...)
result = await spinner.spin("receipt.jpg")
```

---

## Performance Comparison

| Backend | Quality | Speed | CPU | GPU | Setup |
|---------|---------|-------|-----|-----|-------|
| DeepSeek | ⭐⭐⭐⭐⭐ | Fast | Low | High | Complex |
| Tesseract | ⭐⭐⭐⭐ | Fast | Medium | None | Easy |
| Fallback | ⭐ | Instant | None | None | None |

---

## Next Steps

### Immediate (Choose One)

**Option A: Quick Start (Tesseract)** ← Recommended for current hardware
```bash
choco install tesseract
pip install pytesseract
python demos/demo_schema_aware_receipt.py
```

**Option B: Best Quality (DeepSeek)** ← Requires GPU upgrade
```bash
# Install CUDA Toolkit
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.8.5
pip install deepseek-ocr
python demos/demo_schema_aware_receipt.py
```

### After Installation

1. Run demos to test:
   ```bash
   python demos/demo_schema_aware_receipt.py
   python demos/demo_voice_correction.py
   ```

2. Start voice UI:
   ```bash
   cd HoloLoom/web_dashboard
   python voice_correction_server.py
   # Open http://localhost:8001
   ```

3. Process your own receipts:
   ```python
   from HoloLoom.spinningWheel import SchemaAwareReceiptSpinner
   from HoloLoom.memory.graph import KG

   spinner = SchemaAwareReceiptSpinner(yarn_graph=KG())
   result = await spinner.spin("my_receipt.jpg")
   ```

---

## Future Enhancement: Visual Tokens

**Status**: Roadmap planned (see `VISUAL_TOKENS_ROADMAP.md`)

DeepSeek-OCR supports **native visual tokens** for context compression:
- 15x compression vs raw text (2.6x vision + 6x structural)
- Preserves layout/formatting
- Integrates with YarnGraph for lossless reconstruction

**Phases**:
1. **Phase 1-2** (CPU-only): Tesseract + structural tokens (6x compression)
2. **Phase 3+** (GPU required): DeepSeek native vision tokens (15x compression)

See `DEEPSEEK_OCR_INTEGRATION.md` and `VISUAL_TOKENS_ROADMAP.md` for details.

---

## Summary

**Recommended Path**:
1. Install Tesseract (easy, works great)
2. Test with demos
3. Later upgrade to DeepSeek if you have CUDA

**Commands**:
```bash
# Install Tesseract (Windows)
choco install tesseract
pip install pytesseract

# Verify
tesseract --version
python -c "import pytesseract; print('OK')"

# Test
python demos/demo_schema_aware_receipt.py
```

That's it! You'll have production-ready OCR working in minutes. 🚀
