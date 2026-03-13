# 技术细节

## 1. 为什么选择级联架构而非端到端?

### 端到端 (E2E) 的问题

我们最初尝试了端到端蒸馏方案：

```
SeamlessM4T (Teacher) → 小模型 (Student)
```

**失败原因：**
- 训练数据太少 (100 条)
- 模型太小 (8.5M 参数) 难以学习语义
- 蒸馏目标是 mel spectrogram，没有语义监督
- 输出声音"像语音"但内容错误

### 级联架构的优势

```
ASR → MT → TTS
```

1. **每步可控** - 可以看到中间结果 (文本)
2. **模块化** - 可单独升级每个组件
3. **调试容易** - 出错能定位是哪一步
4. **质量有保障** - 使用成熟的模型/API

### 延迟对比

| 方案 | 延迟 | 质量 |
|------|------|------|
| 端到端 (理想) | ~100ms | 需要大量数据 |
| 端到端 (当前) | ~8ms | ❌ 不可用 |
| 级联 (当前) | ~1.2s | ✅ 可用 |

---

## 2. Whisper 模型选择

| 模型 | 参数量 | 大小 | 相对速度 | 英语准确率 |
|------|--------|------|----------|------------|
| tiny | 39M | 75MB | 32x | ~88% |
| base | 74M | 145MB | 16x | ~91% |
| small | 244M | 488MB | 6x | ~94% |
| medium | 769M | 1.5GB | 2x | ~96% |
| large | 1550M | 3.1GB | 1x | ~97% |

**当前选择: tiny**

原因：
- 移动端部署需要小模型
- 清晰语音场景下准确率够用
- 可根据设备性能选择更大模型

---

## 3. 翻译 API 选择

### Google Translate (gtx)

```python
url = "https://translate.googleapis.com/translate_a/single"
params = {
    "client": "gtx",
    "sl": "en",
    "tl": "zh-CN", 
    "dt": "t",
    "q": text
}
```

**优点:**
- 免费，无需 API Key
- 质量高 (Google 级别)
- 延迟低 (~100ms)

**风险:**
- 非官方接口，可能被封
- 有请求频率限制

**备选:**
- MyMemory API - 免费但慢 (~3s)
- DeepL API - 付费，质量高
- 本地模型 - 离线，质量中等

---

## 4. Edge TTS 使用

### 安装

```bash
pip install edge-tts
```

### 基本用法

```python
import edge_tts
import asyncio

async def tts(text, output_path):
    communicate = edge_tts.Communicate(
        text, 
        voice="zh-CN-XiaoxiaoNeural"
    )
    await communicate.save(output_path)

asyncio.run(tts("你好", "output.mp3"))
```

### 可用中文声音

```bash
edge-tts --list-voices | grep zh-CN
```

| Voice | 性别 | 风格 |
|-------|------|------|
| zh-CN-XiaoxiaoNeural | 女 | 活泼、新闻 |
| zh-CN-YunxiNeural | 男 | 标准、叙述 |
| zh-CN-XiaoyiNeural | 女 | 温柔、客服 |
| zh-CN-YunjianNeural | 男 | 运动解说 |
| zh-CN-liaoning-XiaobeiNeural | 女 | 东北方言 |

---

## 5. 音频格式处理

### 浏览器录音格式

浏览器 MediaRecorder 默认输出 WebM (Opus 编码):

```javascript
new MediaRecorder(stream, { mimeType: 'audio/webm' })
```

### Whisper 需要的格式

- 采样率: 16kHz
- 声道: 单声道 (mono)
- 格式: WAV/FLAC/MP3

### FFmpeg 转换命令

```bash
ffmpeg -y -i input.webm -ar 16000 -ac 1 output.wav
```

参数说明:
- `-y` - 覆盖输出文件
- `-ar 16000` - 采样率 16kHz
- `-ac 1` - 单声道

---

## 6. 延迟优化策略

### 当前延迟分解

| 步骤 | 耗时 | 占比 |
|------|------|------|
| FFmpeg 转换 | ~50ms | 4% |
| Whisper ASR | ~600ms | 50% |
| Google MT | ~100ms | 8% |
| Edge TTS | ~400ms | 33% |
| FFmpeg 转WAV | ~50ms | 4% |
| **总计** | ~1200ms | 100% |

### 优化方向

1. **模型预热** - 启动时预加载 Whisper
2. **流式 ASR** - Whisper 支持流式，可边录边转写
3. **并行 TTS** - 翻译完成前开始准备 TTS
4. **本地 MT** - 消除网络延迟

### 理论最优延迟

| 步骤 | 优化后 |
|------|--------|
| 流式 ASR | ~200ms |
| 本地 MT | ~100ms |
| 流式 TTS | ~200ms |
| **总计** | ~500ms |

---

## 7. 移动端部署路径

### 方案 A: React Native + ONNX Runtime

```
┌────────────────┐
│  React Native  │
│  ┌──────────┐  │
│  │ onnxruntime │
│  │  whisper   │
│  │  nmt model │
│  └──────────┘  │
│  + Edge TTS   │
└────────────────┘
```

### 方案 B: 原生 iOS/Android

```
iOS: Core ML + AVFoundation
Android: TensorFlow Lite + MediaPlayer
```

### 方案 C: 混合架构

```
端侧: ASR (Whisper ONNX)
云端: MT + TTS (低延迟 API)
```

---

## 8. 关键代码片段

### 级联翻译核心逻辑

```python
def cascade_translate(input_audio, output_audio):
    # 1. ASR
    model = whisper.load_model("tiny")
    result = model.transcribe(input_audio, language="en")
    text_en = result["text"]
    
    # 2. MT
    text_zh = translate_google(text_en)
    
    # 3. TTS
    asyncio.run(edge_tts.Communicate(
        text_zh, 
        "zh-CN-XiaoxiaoNeural"
    ).save(output_audio))
```

### Web 服务关键点

```javascript
// 解析 multipart 音频
const boundary = contentType.split('boundary=')[1];
const headerEnd = buffer.indexOf('\r\n\r\n');
const audioData = buffer.slice(headerEnd + 4, endBoundary);

// 调用 Python 翻译
spawn('python', ['translate_cascade_fast.py', input, output]);
```

---

## 9. 已知问题和限制

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| Google API 可能被封 | 翻译失败 | 备选 MyMemory |
| Edge TTS 需要网络 | 离线不可用 | 换本地 TTS |
| Whisper 首次加载慢 | 启动延迟 | 预加载 |
| 不支持流式 | 需等录音结束 | 实现 VAD |

---

## 10. 参考资料

- [Whisper 论文](https://arxiv.org/abs/2212.04356)
- [Edge TTS 文档](https://github.com/rany2/edge-tts)
- [SeamlessM4T](https://github.com/facebookresearch/seamless_communication)
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
