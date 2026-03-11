# Test Audio Samples

Pre-generated audio samples for testing the S2ST demo.

## English Samples

| File | Content | Duration |
|------|---------|----------|
| `english_hello.wav` | "Hello, this is a test of the speech to speech translation system. How are you today?" | ~3s |
| `english_weather.wav` | "The weather today is sunny with a high of 25 degrees. It's a perfect day to go outside." | ~4s |
| `english_intro.wav` | "My name is John. I am a software engineer from California. Nice to meet you." | ~4s |

## Chinese Samples (中文样本)

| File | Content | Duration |
|------|---------|----------|
| `chinese_hello.wav` | "你好，这是语音翻译系统的测试。今天你好吗？" | ~3s |
| `chinese_weather.wav` | "今天天气晴朗，最高气温25度。这是一个出门的好日子。" | ~4s |
| `chinese_intro.wav` | "我叫小明，我是一名来自北京的程序员。很高兴认识你。" | ~4s |

## Usage

Upload these files to the demo interface to test translation:

1. Open the demo at http://localhost:7860
2. Select a language pair (e.g., EN→ZH or ZH→EN)
3. Click "Upload Audio File" and select a sample
4. Click "Translate" to process

## Format

All samples are:
- Format: WAV (PCM)
- Sample rate: 22050 Hz
- Channels: Mono
- Bit depth: 16-bit

## Generating Your Own Samples

On macOS, you can use the `say` command:

```bash
# English
say -o output.aiff "Your text here"
afconvert -f WAVE -d LEI16 output.aiff output.wav

# Chinese (Tingting voice)
say -v Tingting -o output.aiff "你的文字"
afconvert -f WAVE -d LEI16 output.aiff output.wav
```
