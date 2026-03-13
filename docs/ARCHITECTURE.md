# S2ST 语音翻译系统架构

## 概述

本项目实现端侧语音到语音翻译 (Speech-to-Speech Translation)，目标是在移动设备上实现低延迟、高质量的实时翻译。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户设备 (手机/电脑)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │  麦克风  │ → │   ASR   │ → │   MT    │ → │   TTS   │ → 🔊 │
│  │ (录音)   │    │ (语音→  │    │ (英→中  │    │ (文本→  │      │
│  └─────────┘    │  文本)   │    │  翻译)  │    │  语音)  │      │
│                 └─────────┘    └─────────┘    └─────────┘      │
│                                                                 │
│  延迟分解:        ~0.6s         ~0.1s         ~0.5s            │
│                                                                 │
│                        总延迟: ~1.2s                            │
└─────────────────────────────────────────────────────────────────┘
```

## 技术栈

### 1. ASR (自动语音识别)

| 组件 | 选择 | 说明 |
|------|------|------|
| 模型 | Whisper Tiny | OpenAI 开源，39MB |
| 延迟 | ~600ms | CPU 推理 |
| 准确率 | 中等 | 适合清晰语音 |

**为什么选 Whisper Tiny?**
- 开源免费，无需 API 费用
- 支持多语言，可扩展到其他语种
- 有更大版本 (base/small/medium) 可选，准确率更高

### 2. MT (机器翻译)

| 组件 | 选择 | 说明 |
|------|------|------|
| 方案 | Google Translate API | 非官方 gtx 接口 |
| 延迟 | ~100ms | 网络请求 |
| 质量 | 高 | Google 翻译质量 |

**备选方案:**
- Helsinki-NLP/opus-mt-en-zh (本地，300MB，质量稍低)
- MyMemory API (免费备选)
- 自训练 NMT 模型 (需要数据)

### 3. TTS (文本转语音)

| 组件 | 选择 | 说明 |
|------|------|------|
| 方案 | Edge TTS | 微软 Azure 免费 TTS |
| 声音 | zh-CN-XiaoxiaoNeural | 中文女声，自然度高 |
| 延迟 | ~500ms | 含网络 + 格式转换 |

**可用声音:**
- `zh-CN-XiaoxiaoNeural` - 女声，活泼
- `zh-CN-YunxiNeural` - 男声，标准
- `zh-CN-XiaoyiNeural` - 女声，温柔

## 数据流

```
1. 用户录音 (WebRTC MediaRecorder)
   ↓ audio/webm
2. 前端发送到后端
   ↓ multipart/form-data
3. FFmpeg 转换格式
   ↓ 16kHz mono WAV
4. Whisper ASR 转写
   ↓ 英文文本
5. Google Translate 翻译
   ↓ 中文文本
6. Edge TTS 合成
   ↓ MP3
7. FFmpeg 转换为 WAV
   ↓ 16kHz mono WAV
8. 返回前端播放
```

## 性能指标

| 指标 | 目标 | 当前 |
|------|------|------|
| 总延迟 | <2s | **1.2s** ✅ |
| 模型大小 | <200MB | ~40MB (Whisper) ✅ |
| 翻译准确率 | >90% | ~95% (Google) ✅ |

## 文件结构

```
s2st-distill/
├── cascade/
│   └── translate_cascade_fast.py   # 级联翻译主逻辑
├── web/
│   └── server.js                   # Node.js Web 服务
├── output/                         # 翻译输出音频
├── test_source.wav                 # 测试用英语音频
├── test_target.wav                 # 参考中文音频
└── docs/
    ├── ARCHITECTURE.md             # 本文件
    └── TECHNICAL_DETAILS.md        # 技术细节
```

## 下一步

1. **移动端部署** - 将模型导出为 ONNX/TFLite
2. **离线翻译** - 替换 Google API 为本地模型
3. **流式处理** - 实现边说边翻译
4. **说话人保持** - 保留原说话人音色
