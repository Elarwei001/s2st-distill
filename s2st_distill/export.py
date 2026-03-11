"""
Export utilities for mobile deployment (ONNX, CoreML, TFLite).
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import os


def export_onnx(
    model: nn.Module,
    output_path: str = "model.onnx",
    audio_length_sec: float = 5.0,
    sample_rate: int = 16000,
    opset_version: int = 14
) -> str:
    """
    Export model to ONNX format.
    
    ONNX is an intermediate format that can be converted to
    CoreML (iOS) or TFLite (Android).
    
    Args:
        model: PyTorch model to export
        output_path: Output file path
        audio_length_sec: Sample audio length for tracing
        sample_rate: Audio sample rate
        opset_version: ONNX opset version
    
    Returns:
        Path to exported model
    """
    model.eval()
    
    # Create sample input
    audio_length = int(sample_rate * audio_length_sec)
    dummy_audio = torch.randn(1, audio_length)
    
    # Move to CPU for export
    model = model.cpu()
    dummy_audio = dummy_audio.cpu()
    
    print(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_audio,
        output_path,
        input_names=["audio"],
        output_names=["translated_audio"],
        dynamic_axes={
            "audio": {0: "batch_size", 1: "audio_length"},
            "translated_audio": {0: "batch_size", 1: "output_length"}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    # Verify export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verified successfully")
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Exported to {output_path} ({size_mb:.1f} MB)")
    
    return output_path


def export_coreml(
    onnx_path: str,
    output_path: str = "model.mlpackage",
    minimum_ios_version: str = "iOS16",
    compute_precision: str = "float16"
) -> str:
    """
    Convert ONNX model to CoreML for iOS deployment.
    
    CoreML models can leverage Apple Neural Engine (ANE)
    for fast on-device inference.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output CoreML package path
        minimum_ios_version: Minimum iOS deployment target
        compute_precision: Compute precision (float16 or float32)
    
    Returns:
        Path to exported model
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install with: pip install coremltools"
        )
    
    print(f"Converting ONNX to CoreML: {output_path}")
    
    # Parse iOS version
    ios_target = getattr(ct.target, minimum_ios_version, ct.target.iOS16)
    
    # Set precision
    precision = ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
    
    # Convert
    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ios_target,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,  # Use all available compute units (including ANE)
    )
    
    # Add metadata
    model.author = "S2ST-Distill"
    model.short_description = "On-device speech-to-speech translation"
    model.version = "1.0.0"
    
    # Save
    model.save(output_path)
    
    size_mb = _get_package_size(output_path)
    print(f"✓ Exported to {output_path} ({size_mb:.1f} MB)")
    
    return output_path


def export_tflite(
    onnx_path: str,
    output_path: str = "model.tflite",
    quantize: bool = True,
    use_float16: bool = True
) -> str:
    """
    Convert ONNX model to TFLite for Android deployment.
    
    TFLite models can use GPU delegate or NNAPI for
    hardware acceleration on Android.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output TFLite file path
        quantize: Apply default optimizations
        use_float16: Use float16 quantization
    
    Returns:
        Path to exported model
    """
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        raise ImportError(
            "tensorflow and onnx-tf are required for TFLite export. "
            "Install with: pip install tensorflow onnx-tf"
        )
    
    print(f"Converting ONNX to TFLite: {output_path}")
    
    # Create temp directory for SavedModel
    saved_model_dir = output_path.replace(".tflite", "_saved_model")
    
    # ONNX → TensorFlow SavedModel
    print("  Step 1/3: ONNX → TensorFlow SavedModel")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)
    
    # TensorFlow → TFLite
    print("  Step 2/3: TensorFlow → TFLite")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Optimization settings
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if use_float16:
        converter.target_spec.supported_types = [tf.float16]
    
    # Enable all ops for compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    print("  Step 3/3: Converting...")
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    # Cleanup temp SavedModel
    import shutil
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Exported to {output_path} ({size_mb:.1f} MB)")
    
    return output_path


def _get_package_size(package_path: str) -> float:
    """Get total size of a directory (for CoreML packages)."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(package_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)


def verify_exported_model(
    model_path: str,
    sample_audio: Optional[torch.Tensor] = None
) -> bool:
    """
    Verify exported model can be loaded and run inference.
    
    Args:
        model_path: Path to exported model
        sample_audio: Optional sample audio for inference test
    
    Returns:
        True if verification passed
    """
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == ".onnx":
        return _verify_onnx(model_path, sample_audio)
    elif ext == ".mlpackage" or model_path.endswith(".mlpackage"):
        return _verify_coreml(model_path)
    elif ext == ".tflite":
        return _verify_tflite(model_path, sample_audio)
    else:
        print(f"Unknown model format: {ext}")
        return False


def _verify_onnx(model_path: str, sample_audio: Optional[torch.Tensor]) -> bool:
    """Verify ONNX model."""
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        if sample_audio is not None:
            audio_np = sample_audio.numpy()
            _ = session.run(None, {input_name: audio_np})
            print("✓ ONNX inference test passed")
        
        print("✓ ONNX model verified")
        return True
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False


def _verify_coreml(model_path: str) -> bool:
    """Verify CoreML model."""
    try:
        import coremltools as ct
        
        model = ct.models.MLModel(model_path)
        print(f"✓ CoreML model verified")
        print(f"  Compute units: {model.compute_unit}")
        return True
    except Exception as e:
        print(f"✗ CoreML verification failed: {e}")
        return False


def _verify_tflite(model_path: str, sample_audio: Optional[torch.Tensor]) -> bool:
    """Verify TFLite model."""
    try:
        import tensorflow as tf
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✓ TFLite model verified")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        
        return True
    except Exception as e:
        print(f"✗ TFLite verification failed: {e}")
        return False
