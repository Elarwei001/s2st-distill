#!/usr/bin/env python3
"""
级联式语音翻译 v2 (EN → ZH)
ASR (Whisper) → MT (本地 Helsinki-NLP) → TTS (Edge-TTS)

优化：使用本地翻译模型，避免网络延迟
"""
import sys
import os
import time
import asyncio
import tempfile
import json
import whisper
import edge_tts
from transformers import MarianMTModel, MarianTokenizer

# 全局缓存
_whisper_model = None
_mt_model = None
_mt_tokenizer = None


def get_whisper_model(model_size="tiny"):
    global _whisper_model
    if _whisper_model is None:
        print("[Cache] Loading Whisper model...", file=sys.stderr)
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def get_mt_model():
    global _mt_model, _mt_tokenizer
    if _mt_model is None:
        print("[Cache] Loading translation model...", file=sys.stderr)
        model_name = "Helsinki-NLP/opus-mt-en-zh"
        _mt_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _mt_model = MarianMTModel.from_pretrained(model_name)
    return _mt_model, _mt_tokenizer


def translate_text_local(text: str) -> str:
    """使用本地 Helsinki-NLP 模型翻译"""
    model, tokenizer = get_mt_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


async def text_to_speech(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    """使用 Edge TTS 生成中文语音"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def cascade_translate(input_audio: str, output_audio: str, whisper_model: str = "tiny") -> dict:
    """级联式语音翻译"""
    result = {
        "input": input_audio,
        "output": output_audio,
        "timings": {},
        "text_en": "",
        "text_zh": "",
    }
    
    total_start = time.time()
    
    # Step 1: ASR
    print("[1/3] ASR: Transcribing...", file=sys.stderr)
    t0 = time.time()
    model = get_whisper_model(whisper_model)
    asr_result = model.transcribe(input_audio, language="en")
    text_en = asr_result["text"].strip()
    result["text_en"] = text_en
    result["timings"]["asr"] = time.time() - t0
    print(f"  → EN: {text_en}", file=sys.stderr)
    
    # Step 2: MT (本地)
    print("[2/3] MT: Translating...", file=sys.stderr)
    t0 = time.time()
    text_zh = translate_text_local(text_en)
    result["text_zh"] = text_zh
    result["timings"]["mt"] = time.time() - t0
    print(f"  → ZH: {text_zh}", file=sys.stderr)
    
    # Step 3: TTS
    print("[3/3] TTS: Generating speech...", file=sys.stderr)
    t0 = time.time()
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    
    asyncio.run(text_to_speech(text_zh, mp3_path))
    
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path, 
        "-ar", "16000", "-ac", "1", 
        output_audio
    ], capture_output=True)
    os.unlink(mp3_path)
    
    result["timings"]["tts"] = time.time() - t0
    result["timings"]["total"] = time.time() - total_start
    
    print(f"\n✅ Done! Total: {result['timings']['total']:.2f}s", file=sys.stderr)
    print(f"   ASR: {result['timings']['asr']:.2f}s | MT: {result['timings']['mt']:.2f}s | TTS: {result['timings']['tts']:.2f}s", file=sys.stderr)
    
    return result


# 预热函数 - 提前加载模型
def warmup():
    print("Warming up models...", file=sys.stderr)
    get_whisper_model("tiny")
    get_mt_model()
    print("Warmup complete!", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: translate_cascade_v2.py <input.wav> <output.wav> [whisper_model]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "tiny"
    
    result = cascade_translate(input_path, output_path, model_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
