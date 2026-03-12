# Modal Training Guide

使用 Modal 云 GPU 训练 S2ST 蒸馏模型。

## 💰 成本估算

| 训练任务 | GPU | 时间 | 费用 |
|----------|-----|------|------|
| 单语言对 (EN→ZH) | A100 | ~1-2h | ~$3-6 |
| 全部 6 语言对 (并行) | A100 x6 | ~2h | ~$15-20 |
| 全部 6 语言对 (串行) | A100 | ~8-12h | ~$25-35 |

## 🚀 Quick Start

### 1. 安装 Modal CLI

```bash
pip install modal
modal setup  # 登录并配置
```

### 2. 训练单语言对

```bash
cd /tmp/s2st-distill

# 训练 EN→ZH
modal run modal_train.py --lang-pair en_zh

# 训练 ZH→EN
modal run modal_train.py --lang-pair zh_en

# 可选参数
modal run modal_train.py --lang-pair en_zh --epochs 5  # 减少轮数
```

### 3. 训练全部语言对（并行，更快）

```bash
modal run modal_train.py --all
```

### 4. 下载训练好的模型

```bash
# 查看可用模型
modal run modal_train.py::download_models

# 下载到本地
modal volume get s2st-models /models/en_zh ./models/en_zh
```

## 📊 支持的语言对

| ID | 方向 | 语言 |
|----|------|------|
| `en_zh` | EN → ZH | 英语 → 中文 |
| `zh_en` | ZH → EN | 中文 → 英语 |
| `en_fr` | EN → FR | 英语 → 法语 |
| `fr_en` | FR → EN | 法语 → 英语 |
| `zh_fr` | ZH → FR | 中文 → 法语 |
| `fr_zh` | FR → ZH | 法语 → 中文 |

## 🔧 高级配置

### 使用 Weights & Biases 追踪实验

```bash
# 创建 wandb secret
modal secret create wandb WANDB_API_KEY=your_key_here

# 带追踪的训练
modal run modal_train.py --lang-pair en_zh --wandb
```

### 使用 H100 GPU（更快，更贵）

编辑 `modal_train.py`，修改：
```python
@app.function(
    gpu="H100",  # 改为 H100
    ...
)
```

H100 价格约 $4.50/hr，速度约快 2x。

## 📁 输出文件

训练完成后，每个语言对会生成：

```
/models/{lang_pair}/
├── model.onnx          # ONNX 模型（用于部署）
├── model.pt            # PyTorch 权重
└── checkpoint_epoch*.pt  # 训练检查点
```

## ❓ 常见问题

### Q: Modal 免费额度是多少？
A: 新用户 $30/月免费额度，足够训练 5-10 个语言对。

### Q: 训练中断了怎么办？
A: 检查点每 5 个 epoch 保存一次，可以从中断处继续。

### Q: 如何查看训练进度？
A: 
```bash
modal app logs s2st-distill
```

### Q: 训练完怎么用？
A: 下载 ONNX 模型，放到 `models/{lang_pair}/model.onnx`，然后运行 demo。
