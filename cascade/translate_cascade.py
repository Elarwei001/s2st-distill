#!/usr/bin/env python3
"""
级联式语音翻译 (EN → ZH)
ASR (Whisper) → MT (深度翻译API/本地) → TTS (Edge-TTS)
"""
import sys
import os
import time
import asyncio
import tempfile
import whisper
import edge_tts

# 翻译 - 使用简单的 API (免费)
# 先试 MyMemory API，免费且无需 key
import urllib.request
import urllib.parse
import json

def translate_text(text: str, src: str = "en", tgt: str = "zh-CN") -> str:
    """使用多个翻译 API，优先选择快的"""
    
    # 方案1: LibreTranslate (通常更快)
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://libretranslate.com/translate",
            data=json.dumps({
                "q": text,
                "source": "en", 
                "target": "zh"
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if "translatedText" in data:
                return data["translatedText"]
    except Exception as e:
        print(f"LibreTranslate error: {e}", file=sys.stderr)
    
    # 方案2: MyMemory 备选
    try:
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair={src}|{tgt}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            if data.get("responseStatus") == 200:
                return data["responseData"]["translatedText"]
    except Exception as e:
        print(f"MyMemory error: {e}", file=sys.stderr)
    
    return f"[翻译失败] {text}"


async def text_to_speech(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    """使用 Edge TTS 生成中文语音"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def cascade_translate(input_audio: str, output_audio: str, whisper_model: str = "tiny") -> dict:
    """
    级联式语音翻译
    
    Returns:
        dict with timing info and intermediate results
    """
    result = {
        "input": input_audio,
        "output": output_audio,
        "timings": {},
        "text_en": "",
        "text_zh": "",
    }
    
    total_start = time.time()
    
    # Step 1: ASR (Speech to Text)
    print("[1/3] ASR: Loading Whisper...", file=sys.stderr)
    t0 = time.time()
    model = whisper.load_model(whisper_model)
    result["timings"]["whisper_load"] = time.time() - t0
    
    print("[1/3] ASR: Transcribing...", file=sys.stderr)
    t0 = time.time()
    asr_result = model.transcribe(input_audio, language="en")
    text_en = asr_result["text"].strip()
    result["text_en"] = text_en
    result["timings"]["asr"] = time.time() - t0
    print(f"  → EN: {text_en}", file=sys.stderr)
    
    # Step 2: MT (Machine Translation)
    print("[2/3] MT: Translating to Chinese...", file=sys.stderr)
    t0 = time.time()
    text_zh = translate_text(text_en)
    result["text_zh"] = text_zh
    result["timings"]["mt"] = time.time() - t0
    print(f"  → ZH: {text_zh}", file=sys.stderr)
    
    # Step 3: TTS (Text to Speech)
    print("[3/3] TTS: Generating speech...", file=sys.stderr)
    t0 = time.time()
    
    # edge-tts 输出 mp3，需要转换为 wav
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    
    asyncio.run(text_to_speech(text_zh, mp3_path))
    
    # 转换为 wav
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: translate_cascade.py <input.wav> <output.wav> [whisper_model]")
        print("  whisper_model: tiny, base, small (default: tiny)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "tiny"
    
    result = cascade_translate(input_path, output_path, model_size)
    
    # 输出 JSON 结果
    print(json.dumps(result, ensure_ascii=False, indent=2))
