# Technical Specification: S2ST Model Distillation

> Complete guide for distilling multilingual speech-to-speech translation models into lightweight single language-pair models for on-device deployment.

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Environment Setup](#phase-1-environment-setup)
3. [Phase 2: Language-Pair Pruning](#phase-2-language-pair-pruning)
4. [Phase 3: Knowledge Distillation](#phase-3-knowledge-distillation)
5. [Phase 4: Layer Pruning](#phase-4-layer-pruning)
6. [Phase 5: Voice Preservation](#phase-5-voice-preservation)
7. [Phase 6: Quantization](#phase-6-quantization)
8. [Phase 7: Mobile Deployment](#phase-7-mobile-deployment)
9. [Phase 8: Evaluation](#phase-8-evaluation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Model Size | 20-50 MB | Downloadable on-demand |
| End-to-End Latency | < 2.5 s | Including algorithmic lookahead |
| Inference Latency | < 300 ms | Neural network computation |
| Voice Naturalness | MOS > 3.5 | Preserve timbre and prosody |
| Translation Quality | BLEU 28+ | Acceptable quality loss |

### Latency Breakdown

```
Total Latency = Algorithmic Delay + Inference Time + Audio I/O

Algorithmic Delay: 1.5-2.0s (model needs context before translating)
Inference Time:    200-300ms (neural network forward pass)
Audio I/O:         50-100ms  (encoding/decoding)
─────────────────────────────────────────────────────────────
Total:             ~2.0-2.5s
```

### Base Model Candidates

| Model | Parameters | Features | Recommendation |
|-------|------------|----------|----------------|
| **SeamlessM4T-Small** | 281M | Official on-device version, full S2ST | ⭐⭐⭐⭐⭐ |
| SimulTron | ~100M | Google's mobile-optimized | ⭐⭐⭐⭐ |
| Translatotron 2 | ~200M | High quality direct S2ST | ⭐⭐⭐ |

**Recommended**: Start with **SeamlessM4T-Small (281M)** as it's already optimized for on-device use.

---

## Phase 1: Environment Setup

### 1.1 System Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 50GB+ disk space

### 1.2 Installation

```bash
# Create virtual environment
conda create -n s2st python=3.10
conda activate s2st

# Install PyTorch with CUDA
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers==4.36.0
pip install fairseq2==0.2.0  # Meta SeamlessM4T dependency
pip install datasets sacrebleu
pip install onnx onnxruntime

# Install mobile export tools
pip install coremltools      # iOS
pip install ai-edge-torch    # Android

# Clone SeamlessM4T
git clone https://github.com/facebookresearch/seamless_communication.git
cd seamless_communication
pip install -e .
```

### 1.3 Download Base Model

```python
from transformers import AutoProcessor, SeamlessM4TModel

# Download Small version (281M parameters)
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-unity-small")
model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-unity-small")

# Verify model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.1f}M")
# Expected: ~281M
```

### 1.4 Prepare Dataset

**Recommended datasets for EN→ZH:**

| Dataset | Size | Description |
|---------|------|-------------|
| CoVoST 2 | 430h | Multilingual speech translation |
| CVSS | 100h | Speech-to-speech with alignments |
| LibriTrans | 236h | Audiobook translations |

```python
from datasets import load_dataset

# Load CoVoST 2 EN→ZH
dataset = load_dataset("facebook/covost2", "en_zh-CN", split="train")

# Data format:
# {
#   "audio": {"array": [...], "sampling_rate": 16000},
#   "sentence": "Hello, how are you?",
#   "translation": "你好，你好吗？"
# }
```

---

## Phase 2: Language-Pair Pruning

### 2.1 Concept

SeamlessM4T supports 100+ languages. For a single language pair (e.g., EN→ZH), we can remove:
- Unused language embeddings
- Unused vocabulary tokens
- Language-specific adapter layers

**Expected size reduction: ~60%**

### 2.2 Implementation

```python
import torch
from transformers import SeamlessM4TModel

class LanguagePairPruner:
    def __init__(self, model, src_lang="eng", tgt_lang="cmn"):
        self.model = model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def get_language_ids(self):
        """Get language IDs for source and target."""
        config = self.model.config
        src_id = config.lang_code_to_id.get(self.src_lang)
        tgt_id = config.lang_code_to_id.get(self.tgt_lang)
        return src_id, tgt_id
    
    def prune_embeddings(self):
        """Remove unused language embeddings."""
        src_id, tgt_id = self.get_language_ids()
        
        # Get original embedding layer
        embed_layer = self.model.text_decoder.embed_tokens
        original_weight = embed_layer.weight.data
        
        # Identify tokens used by target language
        # (This requires language-specific vocabulary analysis)
        used_tokens = self._get_language_tokens(self.tgt_lang)
        
        # Create pruned embedding
        new_vocab_size = len(used_tokens)
        new_embed = torch.nn.Embedding(new_vocab_size, embed_layer.embedding_dim)
        new_embed.weight.data = original_weight[used_tokens]
        
        # Replace embedding layer
        self.model.text_decoder.embed_tokens = new_embed
        
        return self.model
    
    def _get_language_tokens(self, lang):
        """Get vocabulary tokens for a specific language."""
        # Implementation depends on tokenizer
        # Return indices of tokens used in target language
        pass
    
    def freeze_unused_params(self):
        """Freeze parameters not needed for the language pair."""
        for name, param in self.model.named_parameters():
            if self._is_unused_param(name):
                param.requires_grad = False
        return self.model
    
    def _is_unused_param(self, name):
        """Check if parameter is unused for this language pair."""
        # Language-specific logic
        return False

# Usage
pruner = LanguagePairPruner(model, "eng", "cmn")
pruned_model = pruner.prune_embeddings()
pruned_model = pruner.freeze_unused_params()
```

### 2.3 Verification

```python
def verify_pruning(original_model, pruned_model):
    """Verify pruning reduced model size."""
    orig_params = sum(p.numel() for p in original_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    reduction = (1 - pruned_params / orig_params) * 100
    print(f"Original: {orig_params / 1e6:.1f}M")
    print(f"Pruned:   {pruned_params / 1e6:.1f}M")
    print(f"Reduction: {reduction:.1f}%")

verify_pruning(model, pruned_model)
```

---

## Phase 3: Knowledge Distillation

### 3.1 Concept

Use the original large model (teacher) to train the pruned model (student), transferring knowledge through soft targets.

### 3.2 Distillation Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: Softmax temperature for soft targets
            alpha: Weight for soft loss (1-alpha for hard loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft target loss (knowledge distillation)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard target loss (ground truth)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### 3.3 Training Loop

```python
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: str = "cuda"
):
    """
    Train student model with knowledge distillation.
    """
    teacher = teacher.to(device).eval()
    student = student.to(device).train()
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = DistillationLoss(temperature, alpha)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        student.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            
            # Teacher forward (no gradient)
            with torch.no_grad():
                teacher_outputs = teacher(audio)
                teacher_logits = teacher_outputs.logits
            
            # Student forward
            student_outputs = student(audio)
            student_logits = student_outputs.logits
            
            # Compute loss
            loss = criterion(student_logits, teacher_logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        student.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                audio = batch["audio"].to(device)
                labels = batch["labels"].to(device)
                
                student_outputs = student(audio)
                val_loss += F.cross_entropy(student_outputs.logits, labels).item()
        
        val_loss /= len(val_dataloader)
        train_loss /= len(train_dataloader)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student.state_dict(), "best_student.pt")
        
        scheduler.step()
    
    return student
```

---

## Phase 4: Layer Pruning

### 4.1 Concept

Iteratively remove the least important layers based on their impact on validation loss.

Reference: [CULL-MT](https://arxiv.org/abs/2411.06506)

### 4.2 Layer Importance Scoring

```python
import torch
from contextlib import contextmanager

@contextmanager
def temporarily_disable_layer(model, layer_idx):
    """Temporarily disable a layer by replacing it with identity."""
    layer = model.encoder.layers[layer_idx]
    original_forward = layer.forward
    
    # Replace with identity
    layer.forward = lambda x, *args, **kwargs: x
    
    try:
        yield
    finally:
        layer.forward = original_forward

def compute_layer_importance(model, dataloader, num_samples=100, device="cuda"):
    """
    Compute importance score for each layer.
    Higher score = more important (bigger loss increase when removed).
    """
    model = model.to(device).eval()
    importance_scores = {}
    
    # Compute baseline loss
    baseline_loss = evaluate_loss(model, dataloader, num_samples, device)
    
    num_layers = len(model.encoder.layers)
    
    for layer_idx in range(num_layers):
        with temporarily_disable_layer(model, layer_idx):
            layer_loss = evaluate_loss(model, dataloader, num_samples, device)
        
        # Importance = loss increase when layer is disabled
        importance_scores[layer_idx] = layer_loss - baseline_loss
        print(f"Layer {layer_idx}: importance = {importance_scores[layer_idx]:.4f}")
    
    return importance_scores

def evaluate_loss(model, dataloader, num_samples, device):
    """Compute average loss on samples."""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(audio)
            loss = F.cross_entropy(outputs.logits, labels)
            total_loss += loss.item()
            count += 1
    
    return total_loss / count
```

### 4.3 Iterative Pruning

```python
def iterative_layer_pruning(
    model,
    dataloader,
    target_layers: int = 8,
    fine_tune_epochs: int = 2,
    device: str = "cuda"
):
    """
    Iteratively remove layers until reaching target count.
    """
    current_layers = len(model.encoder.layers)
    
    while current_layers > target_layers:
        print(f"\n=== Pruning: {current_layers} -> {current_layers - 1} layers ===")
        
        # Compute importance scores
        importance = compute_layer_importance(model, dataloader, device=device)
        
        # Find least important layer
        least_important = min(importance, key=importance.get)
        print(f"Removing layer {least_important} (importance: {importance[least_important]:.4f})")
        
        # Remove the layer
        model = remove_layer(model, least_important)
        current_layers -= 1
        
        # Fine-tune to recover performance
        model = fine_tune(model, dataloader, epochs=fine_tune_epochs)
    
    return model

def remove_layer(model, layer_idx):
    """Remove a layer from the model."""
    layers = list(model.encoder.layers)
    del layers[layer_idx]
    model.encoder.layers = nn.ModuleList(layers)
    model.config.num_hidden_layers = len(layers)
    return model

def fine_tune(model, dataloader, epochs=2, lr=1e-5):
    """Quick fine-tuning after pruning."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for batch in dataloader:
            audio = batch["audio"].to("cuda")
            labels = batch["labels"].to("cuda")
            
            outputs = model(audio)
            loss = F.cross_entropy(outputs.logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
```

---

## Phase 5: Voice Preservation

### 5.1 Concept

To maintain natural-sounding voice in translated speech:
1. **Speaker Encoder**: Extract speaker identity (timbre)
2. **Prosody Transfer**: Transfer rhythm, pitch, pauses from source

### 5.2 Speaker Encoder

```python
import torch
import torch.nn as nn

class SpeakerEncoder(nn.Module):
    """
    Extract speaker embeddings to preserve voice identity.
    Architecture inspired by ECAPA-TDNN.
    """
    def __init__(self, input_dim=80, embed_dim=256):
        super().__init__()
        
        # 1D CNN for frame-level features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        # Final embedding
        self.fc = nn.Linear(512 * 2, embed_dim)  # mean + std
    
    def forward(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: [B, T, 80] mel features
        Returns:
            speaker_embedding: [B, embed_dim]
        """
        # Transpose for Conv1d: [B, 80, T]
        x = mel_spectrogram.transpose(1, 2)
        
        # Encode
        x = self.encoder(x)  # [B, 512, T]
        x = x.transpose(1, 2)  # [B, T, 512]
        
        # Attention pooling
        weights = torch.softmax(self.attention(x), dim=1)  # [B, T, 1]
        
        # Weighted mean and std
        mean = (x * weights).sum(dim=1)  # [B, 512]
        std = torch.sqrt((((x - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) + 1e-6)
        
        # Concatenate and project
        stats = torch.cat([mean, std], dim=-1)  # [B, 1024]
        embedding = self.fc(stats)  # [B, embed_dim]
        
        return F.normalize(embedding, dim=-1)
```

### 5.3 Prosody Extractor and Transfer

```python
class ProsodyExtractor(nn.Module):
    """
    Extract prosodic features: pitch, duration, energy.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.pitch_encoder = nn.LSTM(hidden_dim, 64, batch_first=True, bidirectional=True)
        self.duration_encoder = nn.LSTM(hidden_dim, 64, batch_first=True, bidirectional=True)
        self.energy_encoder = nn.LSTM(hidden_dim, 64, batch_first=True, bidirectional=True)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, T, hidden_dim]
        Returns:
            prosody_features: dict with pitch, duration, energy
        """
        pitch, _ = self.pitch_encoder(hidden_states)
        duration, _ = self.duration_encoder(hidden_states)
        energy, _ = self.energy_encoder(hidden_states)
        
        return {
            "pitch": pitch,        # [B, T, 128]
            "duration": duration,  # [B, T, 128]
            "energy": energy       # [B, T, 128]
        }


class ProsodyTransfer(nn.Module):
    """
    Transfer prosody from source to target speech.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.prosody_extractor = ProsodyExtractor(hidden_dim)
        
        # Project prosody to match decoder hidden dim
        self.prosody_adapter = nn.Sequential(
            nn.Linear(128 * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, source_features, target_hidden):
        """
        Args:
            source_features: Source encoder features [B, T_src, hidden_dim]
            target_hidden: Target decoder hidden states [B, T_tgt, hidden_dim]
        Returns:
            enhanced_hidden: Target hidden with prosody [B, T_tgt, hidden_dim]
        """
        # Extract source prosody
        prosody = self.prosody_extractor(source_features)
        
        # Combine prosody features
        prosody_combined = torch.cat([
            prosody["pitch"],
            prosody["duration"],
            prosody["energy"]
        ], dim=-1)  # [B, T_src, 384]
        
        # Adapt prosody to target length (simple pooling)
        # In practice, use attention or learned alignment
        prosody_pooled = F.adaptive_avg_pool1d(
            prosody_combined.transpose(1, 2),
            target_hidden.size(1)
        ).transpose(1, 2)  # [B, T_tgt, 384]
        
        # Project and add to target
        prosody_adapted = self.prosody_adapter(prosody_pooled)
        
        return target_hidden + 0.3 * prosody_adapted  # Residual connection
```

### 5.4 High-Quality Vocoder

```python
# Using HiFi-GAN for high-quality waveform synthesis
from speechbrain.inference.vocoders import HIFIGAN

class VocoderWrapper:
    def __init__(self, model_source="speechbrain/tts-hifigan-ljspeech"):
        self.hifigan = HIFIGAN.from_hparams(
            source=model_source,
            savedir="pretrained_models/hifigan"
        )
    
    def synthesize(self, mel_spectrogram):
        """
        Convert mel spectrogram to waveform.
        
        Args:
            mel_spectrogram: [B, T, 80]
        Returns:
            waveform: [B, samples]
        """
        waveform = self.hifigan.decode_batch(mel_spectrogram)
        return waveform

# For smaller model, consider quantized HiFi-GAN (~5MB)
```

---

## Phase 6: Quantization

### 6.1 INT8 Dynamic Quantization

```python
import torch

def quantize_int8(model):
    """
    Apply INT8 dynamic quantization.
    Reduces model size ~4x with minimal quality loss.
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.Conv1d},
        dtype=torch.qint8
    )
    return quantized

# Apply quantization
quantized_model = quantize_int8(student_model)

# Verify size reduction
torch.save(quantized_model.state_dict(), "model_int8.pt")
import os
print(f"Quantized size: {os.path.getsize('model_int8.pt') / 1e6:.1f} MB")
```

### 6.2 INT4 Quantization (More Aggressive)

```python
from transformers import BitsAndBytesConfig

# Using bitsandbytes for INT4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)

quantized_model = SeamlessM4TModel.from_pretrained(
    "path/to/student_model",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 6.3 Quantization-Aware Training (QAT)

```python
import torch.quantization as quant

def prepare_qat(model):
    """Prepare model for quantization-aware training."""
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    return model

def finish_qat(model):
    """Convert QAT model to quantized model."""
    quant.convert(model, inplace=True)
    return model

# QAT workflow
qat_model = prepare_qat(student_model)
qat_model = train_distillation(teacher, qat_model, dataloader, epochs=5)
final_model = finish_qat(qat_model)
```

---

## Phase 7: Mobile Deployment

### 7.1 Export to ONNX

```python
import torch.onnx

def export_onnx(model, output_path="model.onnx", audio_length=5):
    """Export model to ONNX format."""
    model.eval()
    
    # Sample input (5 seconds at 16kHz)
    sample_rate = 16000
    dummy_audio = torch.randn(1, sample_rate * audio_length)
    
    torch.onnx.export(
        model,
        dummy_audio,
        output_path,
        input_names=["audio"],
        output_names=["translated_audio"],
        dynamic_axes={
            "audio": {1: "audio_length"},
            "translated_audio": {1: "output_length"}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"ONNX model exported to {output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified!")

export_onnx(quantized_model)
```

### 7.2 iOS Deployment (CoreML)

```python
import coremltools as ct

def export_coreml(onnx_path, output_path="model.mlpackage"):
    """Convert ONNX to CoreML for iOS."""
    
    # Load and convert
    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine
    )
    
    # Add metadata
    model.author = "S2ST-Distill"
    model.short_description = "On-device speech-to-speech translation"
    model.version = "1.0.0"
    
    # Save
    model.save(output_path)
    print(f"CoreML model saved to {output_path}")

export_coreml("model.onnx")
```

**iOS Swift Integration:**

```swift
import CoreML
import AVFoundation

class S2STTranslator {
    private let model: MLModel
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine
        self.model = try S2STModel(configuration: config).model
    }
    
    func translate(audioBuffer: AVAudioPCMBuffer) async throws -> AVAudioPCMBuffer {
        // Convert to MLMultiArray
        let input = try MLMultiArray(shape: [1, NSNumber(value: audioBuffer.frameLength)], dataType: .float32)
        
        // Run inference
        let output = try await model.prediction(from: S2STModelInput(audio: input))
        
        // Convert output to audio buffer
        return convertToAudioBuffer(output.translated_audio)
    }
}
```

### 7.3 Android Deployment (TFLite)

```python
import tensorflow as tf

def export_tflite(onnx_path, output_path="model.tflite"):
    """Convert ONNX to TFLite for Android."""
    import onnx
    from onnx_tf.backend import prepare
    
    # ONNX → TensorFlow SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("saved_model/")
    
    # TensorFlow → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
    
    # Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Enable all ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")

export_tflite("model.onnx")
```

**Android Kotlin Integration:**

```kotlin
class S2STTranslator(context: Context) {
    private val interpreter: Interpreter
    
    init {
        val options = Interpreter.Options().apply {
            // Use GPU delegate for faster inference
            addDelegate(GpuDelegate())
            numThreads = 4
        }
        
        val modelFile = FileUtil.loadMappedFile(context, "model.tflite")
        interpreter = Interpreter(modelFile, options)
    }
    
    fun translate(audioData: FloatArray): FloatArray {
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()
        
        val inputBuffer = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
        inputBuffer.loadArray(audioData)
        
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        
        interpreter.run(inputBuffer.buffer, outputBuffer.buffer)
        
        return outputBuffer.floatArray
    }
}
```

---

## Phase 8: Evaluation

### 8.1 Translation Quality (BLEU)

```python
from sacrebleu import corpus_bleu

def evaluate_bleu(model, test_dataset, asr_model):
    """
    Evaluate translation quality using BLEU score.
    Requires ASR model to transcribe generated speech.
    """
    predictions = []
    references = []
    
    for sample in tqdm(test_dataset):
        audio = sample["audio"]
        reference = sample["translation"]
        
        # Translate
        translated_audio = model(audio)
        
        # Transcribe generated audio
        predicted_text = asr_model.transcribe(translated_audio)
        
        predictions.append(predicted_text)
        references.append([reference])
    
    bleu = corpus_bleu(predictions, references)
    print(f"BLEU Score: {bleu.score:.2f}")
    
    return bleu

# Target: BLEU 28+
```

### 8.2 Voice Naturalness (MOS)

```python
# Using DNSMOS for automated MOS estimation
# https://github.com/microsoft/DNS-Challenge

def evaluate_mos(audio_samples, sample_rate=16000):
    """
    Evaluate speech naturalness using DNSMOS.
    Scale: 1-5 (higher is better)
    """
    from speechmos import DNSMOS
    
    dnsmos = DNSMOS()
    scores = []
    
    for audio in tqdm(audio_samples):
        result = dnsmos.run(audio, sr=sample_rate)
        scores.append(result["mos"])
    
    mean_mos = np.mean(scores)
    print(f"Mean MOS: {mean_mos:.2f}")
    print(f"  1-2: Bad, 2-3: Poor, 3-4: Fair, 4-5: Good")
    
    return mean_mos

# Target: MOS > 3.5
```

### 8.3 Voice Similarity

```python
from resemblyzer import VoiceEncoder, preprocess_wav

def evaluate_voice_similarity(source_audios, translated_audios):
    """
    Evaluate how well speaker identity is preserved.
    Uses cosine similarity of speaker embeddings.
    """
    encoder = VoiceEncoder()
    
    similarities = []
    
    for src, tgt in tqdm(zip(source_audios, translated_audios)):
        # Preprocess
        src_wav = preprocess_wav(src)
        tgt_wav = preprocess_wav(tgt)
        
        # Extract embeddings
        src_embed = encoder.embed_utterance(src_wav)
        tgt_embed = encoder.embed_utterance(tgt_wav)
        
        # Cosine similarity
        similarity = np.dot(src_embed, tgt_embed)
        similarities.append(similarity)
    
    mean_sim = np.mean(similarities)
    print(f"Voice Similarity: {mean_sim:.3f}")
    print(f"  >0.8: Good, >0.7: Acceptable, <0.7: Poor")
    
    return mean_sim

# Target: > 0.75
```

### 8.4 Latency Benchmark

```python
import time
import numpy as np

def benchmark_latency(model, num_runs=100, audio_duration_sec=5, device="cuda"):
    """
    Measure inference latency.
    """
    model = model.to(device).eval()
    
    # Generate test audio
    sample_rate = 16000
    audio = torch.randn(1, sample_rate * audio_duration_sec).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(audio)
    
    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(audio)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    
    print(f"=== Latency Benchmark ({device}) ===")
    print(f"Audio duration: {audio_duration_sec}s")
    print(f"Mean:  {latencies.mean():.1f} ms")
    print(f"P50:   {np.percentile(latencies, 50):.1f} ms")
    print(f"P95:   {np.percentile(latencies, 95):.1f} ms")
    print(f"P99:   {np.percentile(latencies, 99):.1f} ms")
    print(f"RTF:   {latencies.mean() / (audio_duration_sec * 1000):.3f}x")
    
    return latencies

# Target: Mean < 300ms for 5s audio
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Model too large (>50MB) | Insufficient pruning/quantization | More aggressive layer pruning + INT4 |
| High latency (>500ms) | Too many layers / no NPU | Reduce layers, use NPU delegate |
| Poor translation quality | Over-pruning / under-training | More distillation epochs, higher α |
| Robotic voice | Poor vocoder | Use HiFi-GAN, add prosody transfer |
| Lost speaker identity | Missing speaker encoder | Add speaker embedding conditioning |
| Unnatural rhythm | No prosody transfer | Add prosody extraction/transfer |

### Performance Checklist

```bash
# 1. Check model size
ls -lh model.onnx model.tflite model.mlpackage

# 2. Verify inference latency
python scripts/benchmark.py --model model.onnx --runs 100

# 3. Evaluate translation quality
python scripts/evaluate.py --model model.onnx --metric bleu

# 4. Evaluate voice quality
python scripts/evaluate.py --model model.onnx --metric mos

# 5. Evaluate voice similarity
python scripts/evaluate.py --model model.onnx --metric similarity
```

### Expected Results by Phase

| Phase | Model Size | Latency | BLEU | MOS |
|-------|------------|---------|------|-----|
| Base (281M) | ~1.1 GB | ~800ms | 35+ | 4.0 |
| After lang pruning | ~400 MB | ~500ms | 34+ | 4.0 |
| After layer pruning | ~150 MB | ~300ms | 30+ | 3.8 |
| After INT8 quantization | ~40 MB | ~250ms | 30+ | 3.8 |
| **Final target** | **~30 MB** | **<200ms** | **28+** | **3.5+** |

---

## References

1. [SeamlessM4T](https://github.com/facebookresearch/seamless_communication)
2. [SimulTron](https://arxiv.org/abs/2406.02133)
3. [SeamlessExpressive](https://arxiv.org/abs/2312.05187)
4. [CULL-MT](https://arxiv.org/abs/2411.06506)
5. [Google Real-time S2ST](https://research.google/blog/real-time-speech-to-speech-translation/)
6. [NLLB Pruning](https://github.com/naver/nllb-pruning)

---

*Last updated: 2026-03-11*
