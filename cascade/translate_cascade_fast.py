#!/usr/bin/env python3
"""
级联式语音翻译 - 快速版
ASR (Whisper tiny) → MT (Google Translate 非官方) → TTS (Edge-TTS)

针对 Demo 优化：使用最快的免费方案
"""
import sys
import os
import time
import asyncio
import tempfile
import json
import whisper
import edge_tts
import urllib.request
import urllib.parse

# 全局缓存
_whisper_model = None


def get_whisper_model(model_size="tiny"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def translate_google(text: str) -> str:
    """使用 Google Translate 网页版 API (非官方但快)"""
    try:
        # 使用 Google 翻译的内部 API
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "zh-CN", 
            "dt": "t",
            "q": text
        }
        full_url = url + "?" + urllib.parse.urlencode(params)
        
        req = urllib.request.Request(full_url, headers={
            "User-Agent": "Mozilla/5.0"
        })
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            # 提取翻译结果
            translated = ""
            for item in data[0]:
                if item[0]:
                    translated += item[0]
            return translated
    except Exception as e:
        print(f"Google Translate error: {e}", file=sys.stderr)
    
    # 备选: MyMemory
    try:
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair=en|zh-CN"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("responseStatus") == 200:
                return data["responseData"]["translatedText"]
    except Exception as e:
        print(f"MyMemory error: {e}", file=sys.stderr)
    
    return text  # 失败时返回原文


async def text_to_speech(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    """Edge TTS 中文语音"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def cascade_translate(input_audio: str, output_audio: str, whisper_model: str = "tiny") -> dict:
    """级联翻译主函数"""
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
    
    # MT
    t0 = time.time()
    text_zh = translate_google(text_en)
    result["text_zh"] = text_zh
    result["timings"]["mt"] = time.time() - t0
    print(f"MT ({result['timings']['mt']:.2f}s): {text_zh}", file=sys.stderr)
    
    # TTS
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    
    asyncio.run(text_to_speech(text_zh, mp3_path))
    
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", output_audio], 
                   capture_output=True)
    os.unlink(mp3_path)
    
    result["timings"]["tts"] = time.time() - t0
    result["timings"]["total"] = time.time() - total_start
    
    print(f"TTS ({result['timings']['tts']:.2f}s) → Total: {result['timings']['total']:.2f}s", file=sys.stderr)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: translate_cascade_fast.py <input.wav> <output.wav>")
        sys.exit(1)
    
    result = cascade_translate(sys.argv[1], sys.argv[2])
    print(json.dumps(result, ensure_ascii=False))
