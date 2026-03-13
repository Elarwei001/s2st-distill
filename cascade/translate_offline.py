#!/usr/bin/env python3
"""
全离线语音翻译 (EN → ZH)
ASR (Whisper) → MT (Helsinki-NLP 本地) → TTS (系统内置)

无需网络连接
"""
import sys
import os
import time
import json
import whisper
import pyttsx3
import soundfile as sf
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

# 全局缓存
_whisper_model = None
_mt_model = None
_mt_tokenizer = None
_tts_engine = None


def get_whisper_model(model_size="tiny"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def get_mt_model():
    global _mt_model, _mt_tokenizer
    if _mt_model is None:
        model_name = "Helsinki-NLP/opus-mt-en-zh"
        _mt_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _mt_model = MarianMTModel.from_pretrained(model_name)
    return _mt_model, _mt_tokenizer


def get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        # 设置中文声音
        voices = _tts_engine.getProperty('voices')
        for v in voices:
            if 'Tingting' in v.name:  # 中文普通话女声
                _tts_engine.setProperty('voice', v.id)
                break
        _tts_engine.setProperty('rate', 180)  # 语速
    return _tts_engine


def translate_local(text: str) -> str:
    """本地翻译"""
    model, tokenizer = get_mt_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def tts_to_file(text: str, output_path: str):
    """使用系统 TTS 生成音频文件"""
    engine = get_tts_engine()
    # pyttsx3 保存为 aiff，需要转换
    temp_path = output_path.replace('.wav', '_temp.aiff')
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    
    # 转换为 wav
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', temp_path, '-ar', '16000', '-ac', '1', output_path], 
                   capture_output=True)
    try:
        os.unlink(temp_path)
    except:
        pass


def cascade_translate_offline(input_audio: str, output_audio: str, whisper_model: str = "tiny") -> dict:
    """全离线级联翻译"""
    result = {"timings": {}}
    total_start = time.time()
    
    # ASR
    t0 = time.time()
    model = get_whisper_model(whisper_model)
    asr_result = model.transcribe(input_audio, language="en", fp16=False)
    text_en = asr_result["text"].strip()
    result["text_en"] = text_en
    result["timings"]["asr"] = time.time() - t0
    print(f"ASR ({result['timings']['asr']:.2f}s): {text_en}", file=sys.stderr)
    
    # MT (本地)
    t0 = time.time()
    text_zh = translate_local(text_en)
    result["text_zh"] = text_zh
    result["timings"]["mt"] = time.time() - t0
    print(f"MT ({result['timings']['mt']:.2f}s): {text_zh}", file=sys.stderr)
    
    # TTS (系统)
    t0 = time.time()
    tts_to_file(text_zh, output_audio)
    result["timings"]["tts"] = time.time() - t0
    print(f"TTS ({result['timings']['tts']:.2f}s)", file=sys.stderr)
    
    result["timings"]["total"] = time.time() - total_start
    print(f"Total: {result['timings']['total']:.2f}s", file=sys.stderr)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: translate_offline.py <input.wav> <output.wav>")
        sys.exit(1)
    
    result = cascade_translate_offline(sys.argv[1], sys.argv[2])
    print(json.dumps(result, ensure_ascii=False))
