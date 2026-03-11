"""
S2ST-Distill: Distill multilingual S2ST models for on-device deployment.
"""

__version__ = "0.1.0"

from .distiller import S2STDistiller
from .pruning import LanguagePairPruner, LayerPruner
from .voice_preserve import SpeakerEncoder, ProsodyTransfer
from .quantize import quantize_int8, quantize_int4
from .export import export_onnx, export_coreml, export_tflite

__all__ = [
    "S2STDistiller",
    "LanguagePairPruner",
    "LayerPruner",
    "SpeakerEncoder",
    "ProsodyTransfer",
    "quantize_int8",
    "quantize_int4",
    "export_onnx",
    "export_coreml",
    "export_tflite",
]
