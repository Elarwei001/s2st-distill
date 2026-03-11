# S2ST-Distill

> Distill multilingual speech-to-speech translation models into lightweight single language-pair models for on-device real-time inference.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Goal

Take a large multilingual S2ST model (e.g., SeamlessM4T with 100+ languages) and distill it into a **tiny single language-pair model** (e.g., EN→ZH only) that can:

- Run on mobile devices (iOS/Android)
- Achieve **< 2.5s end-to-end latency** for real-time interpretation
- Maintain **natural-sounding voice** with preserved speaker timbre and prosody
- Fit in **20-50MB** for on-demand download

## ✨ Features

- **Language-pair pruning**: Remove unnecessary language embeddings and parameters
- **Knowledge distillation**: Transfer knowledge from teacher to compact student model
- **Layer pruning**: Iteratively remove unimportant layers based on importance scores
- **Voice preservation**: Maintain speaker identity and prosody in translated speech
- **Quantization**: INT8/INT4 quantization for smaller model size
- **Mobile deployment**: Export to CoreML (iOS) and TFLite (Android)

## 📊 Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Model Size | 20-50 MB | Single language-pair, quantized |
| Inference Latency | < 300 ms | Neural network computation only |
| End-to-End Latency | < 2.5 s | Including algorithmic lookahead |
| Translation Quality | BLEU 28+ | Compared to reference translations |
| Voice Naturalness | MOS 3.5+ | Mean Opinion Score (1-5 scale) |
| Voice Similarity | > 0.75 | Cosine similarity of speaker embeddings |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Elarwei001/s2st-distill.git
cd s2st-distill

# Create virtual environment
conda create -n s2st python=3.10
conda activate s2st

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from s2st_distill import S2STDistiller

# Initialize distiller with base model
distiller = S2STDistiller(
    base_model="facebook/seamless-m4t-unity-small",
    source_lang="eng",
    target_lang="cmn"
)

# Run distillation pipeline
student_model = distiller.distill(
    train_dataset="path/to/train.json",
    num_epochs=10,
    target_size_mb=30
)

# Export for mobile
distiller.export_coreml("model.mlpackage")  # iOS
distiller.export_tflite("model.tflite")      # Android
```

## 📁 Project Structure

```
s2st-distill/
├── docs/                    # Documentation
│   ├── TECHNICAL_SPEC.md    # Detailed technical specification
│   ├── ARCHITECTURE.md      # Model architecture overview
│   └── DEPLOYMENT.md        # Mobile deployment guide
├── s2st_distill/            # Main package
│   ├── __init__.py
│   ├── distiller.py         # Main distillation pipeline
│   ├── pruning.py           # Language and layer pruning
│   ├── voice_preserve.py    # Speaker/prosody preservation
│   ├── quantize.py          # Quantization utilities
│   └── export.py            # Mobile export (CoreML/TFLite)
├── scripts/                 # Utility scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── benchmark.py         # Latency benchmark
├── tests/                   # Unit tests
├── examples/                # Example notebooks
├── requirements.txt
├── setup.py
└── README.md
```

## 📖 Documentation

- [Technical Specification](docs/TECHNICAL_SPEC.md) - Detailed implementation guide
- [Architecture Overview](docs/ARCHITECTURE.md) - Model architecture and design decisions
- [Deployment Guide](docs/DEPLOYMENT.md) - iOS and Android deployment instructions

## 🔬 How It Works

### 1. Language-Pair Pruning
Remove embeddings and parameters for languages not in the target pair, reducing model size by ~60%.

### 2. Knowledge Distillation
Use the original model as teacher to train a smaller student model, preserving translation quality.

### 3. Layer Pruning
Iteratively remove least important layers based on importance scores computed from validation loss.

### 4. Voice Preservation
- **Speaker Encoder**: Extract speaker embeddings to preserve voice identity
- **Prosody Transfer**: Transfer pitch, duration, and energy patterns from source to target

### 5. Quantization
Apply INT8/INT4 quantization to further reduce model size with minimal quality loss.

## 📚 References

- [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) - Meta's multilingual S2ST model
- [SimulTron](https://arxiv.org/abs/2406.02133) - Google's on-device simultaneous S2ST
- [CULL-MT](https://arxiv.org/abs/2411.06506) - Language and layer pruning for MT
- [SeamlessExpressive](https://arxiv.org/abs/2312.05187) - Expressive speech translation

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Meta AI for SeamlessM4T
- Google Research for SimulTron and real-time S2ST research
- The open-source speech processing community
