"""
Model quantization utilities for size reduction.
"""

import torch
import torch.nn as nn
from typing import Optional, Set


def quantize_int8(model: nn.Module) -> nn.Module:
    """
    Apply INT8 dynamic quantization.
    
    Reduces model size ~4x with minimal quality loss.
    Best for inference on CPU.
    
    Args:
        model: PyTorch model to quantize
    
    Returns:
        Quantized model
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d, nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized


def quantize_int4(model: nn.Module, device: str = "cuda") -> nn.Module:
    """
    Apply INT4 quantization using bitsandbytes.
    
    More aggressive compression (~8x size reduction).
    Requires bitsandbytes library.
    
    Args:
        model: PyTorch model to quantize
        device: Target device
    
    Returns:
        Quantized model
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for INT4 quantization. "
            "Install with: pip install bitsandbytes"
        )
    
    # Replace Linear layers with INT4 versions
    def replace_linear_with_int4(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Create INT4 linear layer
                int4_linear = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4"
                )
                int4_linear.weight.data = child.weight.data
                if child.bias is not None:
                    int4_linear.bias.data = child.bias.data
                
                setattr(module, name, int4_linear)
            else:
                replace_linear_with_int4(child)
    
    replace_linear_with_int4(model)
    
    return model.to(device)


def prepare_qat(model: nn.Module) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training (QAT).
    
    This allows training with simulated quantization for better
    accuracy after final quantization.
    
    Args:
        model: Model to prepare for QAT
    
    Returns:
        QAT-ready model
    """
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    return model


def finish_qat(model: nn.Module) -> nn.Module:
    """
    Convert QAT model to final quantized model.
    
    Call this after QAT training is complete.
    
    Args:
        model: QAT-trained model
    
    Returns:
        Fully quantized model
    """
    model.eval()
    torch.quantization.convert(model, inplace=True)
    
    return model


def get_model_size_mb(model: nn.Module, quantized: bool = False) -> float:
    """
    Estimate model size in megabytes.
    
    Args:
        model: PyTorch model
        quantized: Whether model is quantized (affects byte calculation)
    
    Returns:
        Estimated size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate bytes per parameter
    bytes_per_param = 1 if quantized else 4  # INT8 vs FP32
    
    size_bytes = total_params * bytes_per_param
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def save_quantized_model(
    model: nn.Module,
    path: str,
    include_optimizer: bool = False
) -> float:
    """
    Save quantized model and return actual file size.
    
    Args:
        model: Quantized model to save
        path: Output file path
        include_optimizer: Whether to include optimizer state
    
    Returns:
        File size in MB
    """
    import os
    
    if include_optimizer:
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), path)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Saved quantized model to {path} ({size_mb:.1f} MB)")
    
    return size_mb
