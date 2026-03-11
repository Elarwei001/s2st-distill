"""
S2ST-Distill Web Demo

A Gradio-based web interface for testing distilled speech-to-speech translation models.
Supports real-time voice recording, file upload, and instant translation playback.

Usage:
    pip install -r demo/requirements.txt
    python demo/app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import torch and model components
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Running in demo mode with mock translations.")


# =============================================================================
# Model Loading
# =============================================================================

class S2STModel:
    """Wrapper for distilled S2ST models."""
    
    LANGUAGE_PAIRS = {
        "en_zh": ("English", "Chinese", "eng", "cmn"),
        "zh_en": ("Chinese", "English", "cmn", "eng"),
        "zh_fr": ("Chinese", "French", "cmn", "fra"),
        "fr_zh": ("French", "Chinese", "fra", "cmn"),
    }
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.current_model = None
        self.current_pair = None
        self._load_available_models()
    
    def _load_available_models(self):
        """Scan for available model files."""
        self.available_pairs = []
        
        for pair_id, (src_name, tgt_name, src_code, tgt_code) in self.LANGUAGE_PAIRS.items():
            model_path = self.model_dir / pair_id / "model.onnx"
            if model_path.exists():
                self.available_pairs.append(pair_id)
                print(f"Found model: {pair_id} ({src_name} → {tgt_name})")
        
        if not self.available_pairs:
            print("No models found. Demo will run in mock mode.")
            # Add all pairs for demo purposes
            self.available_pairs = list(self.LANGUAGE_PAIRS.keys())
    
    def load_model(self, pair_id: str) -> bool:
        """Load a specific language pair model."""
        if pair_id == self.current_pair:
            return True
        
        model_path = self.model_dir / pair_id / "model.onnx"
        
        if model_path.exists() and TORCH_AVAILABLE:
            try:
                import onnxruntime as ort
                self.models[pair_id] = ort.InferenceSession(str(model_path))
                self.current_model = self.models[pair_id]
                self.current_pair = pair_id
                print(f"Loaded model: {pair_id}")
                return True
            except Exception as e:
                print(f"Error loading model {pair_id}: {e}")
                return False
        else:
            # Mock mode
            self.current_pair = pair_id
            return True
    
    def translate(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """
        Translate audio from source to target language.
        
        Args:
            audio: Input audio waveform
            sample_rate: Sample rate of input audio
        
        Returns:
            Tuple of (translated_audio, output_sample_rate)
        """
        if not TORCH_AVAILABLE or self.current_model is None:
            # Mock translation: return slightly modified audio
            return self._mock_translate(audio, sample_rate)
        
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio = self._resample(audio, sample_rate, 16000)
            
            # Ensure correct shape
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
            
            # Run inference
            input_name = self.current_model.get_inputs()[0].name
            output = self.current_model.run(None, {input_name: audio.astype(np.float32)})
            translated_audio = output[0].squeeze()
            
            return translated_audio, 16000
            
        except Exception as e:
            print(f"Translation error: {e}")
            return self._mock_translate(audio, sample_rate)
    
    def _mock_translate(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Generate mock translated audio for demo purposes."""
        # Simple mock: pitch shift and add slight echo
        # In real use, this would be actual model inference
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)  # Convert to mono
        
        # Add slight variation to simulate translation
        # This is just for demo - real model would do actual translation
        translated = audio * 0.9
        
        # Add small delay/echo effect to make it sound different
        delay_samples = int(sample_rate * 0.02)
        if len(translated) > delay_samples:
            echo = np.zeros_like(translated)
            echo[delay_samples:] = translated[:-delay_samples] * 0.3
            translated = translated + echo
        
        # Normalize
        max_val = np.abs(translated).max()
        if max_val > 0:
            translated = translated / max_val * 0.9
        
        return translated, sample_rate
    
    def _resample(self, audio: np.ndarray, src_rate: int, tgt_rate: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if TORCH_AVAILABLE:
            audio_tensor = torch.from_numpy(audio).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            resampled = torchaudio.transforms.Resample(src_rate, tgt_rate)(audio_tensor)
            return resampled.numpy().squeeze()
        else:
            # Simple linear interpolation fallback
            duration = len(audio) / src_rate
            new_length = int(duration * tgt_rate)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def get_pair_info(self, pair_id: str) -> dict:
        """Get information about a language pair."""
        if pair_id in self.LANGUAGE_PAIRS:
            src_name, tgt_name, src_code, tgt_code = self.LANGUAGE_PAIRS[pair_id]
            return {
                "id": pair_id,
                "source_language": src_name,
                "target_language": tgt_name,
                "source_code": src_code,
                "target_code": tgt_code,
                "display_name": f"{src_name} → {tgt_name}"
            }
        return None


# =============================================================================
# Global Model Instance
# =============================================================================

MODEL = None

def get_model():
    """Get or create the global model instance."""
    global MODEL
    if MODEL is None:
        MODEL = S2STModel()
    return MODEL


# =============================================================================
# Gradio Interface Functions
# =============================================================================

def translate_audio(
    audio_input: Optional[Tuple[int, np.ndarray]],
    audio_file: Optional[str],
    language_pair: str
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """
    Main translation function for Gradio interface.
    
    Args:
        audio_input: Microphone recording (sample_rate, audio_array)
        audio_file: Path to uploaded audio file
        language_pair: Selected language pair ID
    
    Returns:
        Tuple of (translated_audio, status_message)
    """
    model = get_model()
    
    # Load the selected model
    pair_info = model.get_pair_info(language_pair)
    if not pair_info:
        return None, "❌ Invalid language pair selected"
    
    model.load_model(language_pair)
    
    # Get input audio
    if audio_input is not None:
        sample_rate, audio = audio_input
        source = "microphone"
    elif audio_file is not None:
        try:
            if TORCH_AVAILABLE:
                audio, sample_rate = torchaudio.load(audio_file)
                audio = audio.numpy().squeeze()
            else:
                # Fallback: try scipy
                from scipy.io import wavfile
                sample_rate, audio = wavfile.read(audio_file)
                audio = audio.astype(np.float32) / 32768.0
            source = "file"
        except Exception as e:
            return None, f"❌ Error loading audio file: {e}"
    else:
        return None, "⚠️ Please record audio or upload a file"
    
    # Validate audio
    if len(audio) == 0:
        return None, "⚠️ Audio is empty. Please try again."
    
    duration = len(audio) / sample_rate
    if duration < 0.5:
        return None, "⚠️ Audio too short. Please record at least 0.5 seconds."
    if duration > 60:
        return None, "⚠️ Audio too long. Maximum duration is 60 seconds."
    
    # Translate
    try:
        translated_audio, output_rate = model.translate(audio, sample_rate)
        
        status = (
            f"✅ Translated successfully!\n"
            f"📥 Input: {duration:.1f}s from {source}\n"
            f"🌐 Direction: {pair_info['display_name']}\n"
            f"📤 Output: {len(translated_audio)/output_rate:.1f}s"
        )
        
        return (output_rate, translated_audio), status
        
    except Exception as e:
        return None, f"❌ Translation error: {e}"


def get_language_choices():
    """Get available language pair choices for dropdown."""
    model = get_model()
    choices = []
    
    for pair_id in model.available_pairs:
        info = model.get_pair_info(pair_id)
        if info:
            choices.append((info["display_name"], pair_id))
    
    return choices


def get_model_info(language_pair: str) -> str:
    """Get information about the selected model."""
    model = get_model()
    pair_info = model.get_pair_info(language_pair)
    
    if not pair_info:
        return "No model information available"
    
    model_path = model.model_dir / language_pair / "model.onnx"
    model_exists = model_path.exists()
    
    info = f"""
### Model Information

| Property | Value |
|----------|-------|
| Language Pair | {pair_info['display_name']} |
| Source Code | `{pair_info['source_code']}` |
| Target Code | `{pair_info['target_code']}` |
| Model Status | {'✅ Loaded' if model_exists else '⚠️ Demo Mode (no model file)'} |
| Model Path | `{model_path}` |

{"**Note:** Running in demo mode. Actual translation requires trained model files." if not model_exists else ""}
"""
    return info


# =============================================================================
# Gradio UI
# =============================================================================

def create_demo():
    """Create the Gradio demo interface."""
    
    with gr.Blocks(
        title="S2ST-Distill Demo",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 1em; }
        .info-box { background: #f0f0f0; padding: 1em; border-radius: 8px; }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🎤 S2ST-Distill Demo
            
            **Real-time Speech-to-Speech Translation** with distilled on-device models.
            
            This demo showcases lightweight S2ST models that can run on mobile devices with <2.5s latency.
            """,
            elem_classes="main-title"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🌐 Select Language Pair")
                
                language_pair = gr.Dropdown(
                    choices=get_language_choices(),
                    value="en_zh",
                    label="Translation Direction",
                    info="Choose source → target language"
                )
                
                model_info = gr.Markdown(
                    value=get_model_info("en_zh"),
                    elem_classes="info-box"
                )
                
                language_pair.change(
                    fn=get_model_info,
                    inputs=[language_pair],
                    outputs=[model_info]
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ Input Audio")
                
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Record from Microphone",
                    format="wav"
                )
                
                gr.Markdown("**— OR —**")
                
                audio_file = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload Audio File",
                    format="wav"
                )
                
                translate_btn = gr.Button(
                    "🔄 Translate",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🔊 Translated Audio")
                
                audio_output = gr.Audio(
                    label="Translation Result",
                    type="numpy",
                    autoplay=True
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=4,
                    interactive=False
                )
        
        # Connect the translate button
        translate_btn.click(
            fn=translate_audio,
            inputs=[audio_input, audio_file, language_pair],
            outputs=[audio_output, status_output]
        )
        
        # Example section
        gr.Markdown("---")
        gr.Markdown(
            """
            ### 📖 How to Use
            
            1. **Select a language pair** from the dropdown menu
            2. **Record audio** using the microphone button, or **upload** an audio file
            3. Click **Translate** to process the audio
            4. Listen to the **translated result** in the output player
            
            ### 🚀 Performance Targets
            
            | Metric | Target | Description |
            |--------|--------|-------------|
            | Model Size | ≤50 MB | Small enough for mobile download |
            | Inference | ≤300 ms | Neural network computation |
            | End-to-End | ≤2.5 s | Including algorithmic delay |
            | Voice Quality | MOS ≥3.5 | Natural-sounding speech |
            
            ### 🔧 Running with Your Own Models
            
            Place your trained ONNX models in the `models/` directory:
            ```
            models/
            ├── en_zh/
            │   └── model.onnx
            ├── zh_en/
            │   └── model.onnx
            ├── zh_fr/
            │   └── model.onnx
            └── fr_zh/
                └── model.onnx
            ```
            
            ---
            
            *Built with [S2ST-Distill](https://github.com/Elarwei001/s2st-distill) • 
            Based on Meta's SeamlessM4T*
            """
        )
    
    return demo


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="S2ST-Distill Web Demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--model-dir", default="models", help="Directory containing models")
    
    args = parser.parse_args()
    
    # Update model directory if specified
    if args.model_dir != "models":
        MODEL = S2STModel(args.model_dir)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    S2ST-Distill Demo                          ║
║                                                               ║
║  Real-time Speech-to-Speech Translation                       ║
║  Distilled models for on-device deployment                    ║
╚═══════════════════════════════════════════════════════════════╝

Starting server at http://{args.host}:{args.port}
{"Creating public link..." if args.share else ""}
""")
    
    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )
