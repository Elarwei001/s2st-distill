# S2ST-Distill Web Demo

A Gradio-based web interface for testing distilled speech-to-speech translation models.

## Quick Start

```bash
# Install dependencies
pip install -r demo/requirements.txt

# Run the demo
python demo/app.py

# Open http://localhost:7860 in your browser
```

## Features

- 🎙️ **Record Audio**: Use your microphone to record speech
- 📁 **Upload Files**: Support for WAV/MP3 audio files
- 🌐 **Multiple Languages**: EN↔ZH, ZH↔FR translation pairs
- 🔊 **Auto Playback**: Hear translations immediately
- 📊 **Real-time Status**: See translation progress and metrics

## Usage

1. **Select a language pair** from the dropdown
2. **Record audio** or **upload a file**
3. Click **Translate**
4. Listen to the translated result!

## Running with Trained Models

Place your ONNX models in the `models/` directory:

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

Then run the demo:

```bash
python demo/app.py --model-dir models
```

## Command Line Options

```
--host       Host to bind to (default: 127.0.0.1)
--port       Port to listen on (default: 7860)
--share      Create a public Gradio link
--model-dir  Directory containing trained models
```

## Demo Mode

If no trained models are found, the demo runs in **mock mode**:
- UI is fully functional
- Audio processing works (record, upload, playback)
- "Translation" returns slightly modified audio (for UI testing)

This allows you to test the interface before training models.

## Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│                   🎤 S2ST-Distill Demo                      │
├─────────────────────────────────────────────────────────────┤
│  Language Pair: [English → Chinese ▼]                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ 🎙️ Record       │    │ 🔊 Translated Audio             │ │
│  │                 │    │                                 │ │
│  │  [● Record]     │    │  [▶ Play]                       │ │
│  │                 │    │                                 │ │
│  │ — OR —          │    │ Status:                         │ │
│  │                 │    │ ✅ Translated successfully!     │ │
│  │ 📁 Upload       │    │ 📥 Input: 3.2s from microphone  │ │
│  │ [Choose File]   │    │ 🌐 Direction: English → Chinese │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│                                                             │
│  [🔄 Translate]                                             │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- Modern web browser with microphone access
- For full functionality: PyTorch + ONNX Runtime

## License

MIT License - see [LICENSE](../LICENSE) for details.
