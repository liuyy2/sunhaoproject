# CIFAR-10：LoRA 半监督学习与锚图谱聚类

本项目在 CIFAR-10 上实现了一条可复现的三阶段实验流水线：

1. 使用全部训练图像进行 SimSiam 风格的自监督学习，只训练 ViT 后若干层中的 LoRA 参数；
2. 使用固定的 1% 标签（500 张、每类 50 张）和其余无标签样本进行半监督训练；
3. 从测试集提取无泄漏特征，并使用分类头、K-Means 和 Anchor Graph Spectral Clustering 评估。

当前主实验使用 `vit_base_patch16_224`、LoRA rank 16、随机种子 0。固定数据划分保存在
`splits/cifar10_1pct_seed0.json`，已完成实验的指标保存在
`results/stage2_seed0_test.json`。

## 当前结果

| 方法 | ACC | NMI | ARI |
| --- | ---: | ---: | ---: |
| EMA 分类头 | 0.9796 | 0.9506 | 0.9558 |
| K-Means（backbone 特征） | 0.9781 | 0.9472 | 0.9527 |
| K-Means（logits） | 0.9791 | 0.9500 | 0.9548 |
| K-Means（probabilities） | 0.9796 | 0.9506 | 0.9558 |
| Anchor Graph（backbone 特征） | 0.8562 | 0.9200 | 0.8489 |

这些结果对应 10,000 张 CIFAR-10 测试图像。测试标签只用于最终计算指标，不参与训练或特征生成。

## 环境准备

建议使用带 CUDA 的 Linux 或 Windows 环境。完整训练使用 ViT-B/16，CPU 可以运行检查，但不适合正式训练。

```bash
git clone https://github.com/liuyy2/sunhaoproject.git
cd sunhaoproject

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

项目通过 `torchvision.datasets.CIFAR10` 读取数据。首次运行可添加 `--download`：

```bash
python prepare_protocol.py --download
```

默认数据目录为 `./CIFAR-10/data`。数据集、模型权重和特征缓存体积较大，不保存在 Git 仓库中。

## 完整复现实验

以下命令都应在仓库根目录执行。

### 0. 生成或核对固定划分

```bash
python prepare_protocol.py \
  --data-dir ./CIFAR-10/data \
  --split-path ./splits/cifar10_1pct_seed0.json \
  --seed 0 \
  --download
```

如果划分文件已经存在，脚本会核对其内容，不会悄悄替换实验协议。

### 1. 全量无标签自监督训练

```bash
python train_stage1_ssl.py \
  --data-dir ./CIFAR-10/data \
  --output-dir ./weights/stage1_seed0 \
  --seed 0 \
  --grad-checkpointing
```

默认训练 100 个 epoch，最新检查点写入 `weights/stage1_seed0/latest.pth`。显存不足时可减小
`--batch-size` 或增大 `--grad-accum`；正式对比实验中请记录参数变化。

### 2. 1% 标签半监督训练

```bash
python train_stage2_semi.py \
  --data-dir ./CIFAR-10/data \
  --split-path ./splits/cifar10_1pct_seed0.json \
  --stage1-checkpoint ./weights/stage1_seed0/latest.pth \
  --output-dir ./weights/stage2_seed0 \
  --seed 0
```

该阶段组合监督对比损失、FixMatch 式伪标签和 EMA teacher。默认训练 100 个 epoch，输出
`weights/stage2_seed0/latest.pth`。中断后可通过 `--resume` 继续：

```bash
python train_stage2_semi.py --resume ./weights/stage2_seed0/latest.pth
```

### 3. 提取测试特征

```bash
python extract_stage3_features.py \
  --checkpoint ./weights/stage2_seed0/latest.pth \
  --data-dir ./CIFAR-10/data \
  --split test \
  --weights ema \
  --output ./features/stage2_seed0_test.npz
```

可使用 `--max-samples 32 --num-workers 0` 做快速冒烟检查；正式评估时必须省略
`--max-samples`，确保包含全部 10,000 张测试图像。

### 4. 聚类与分类评估

```bash
python evaluate_stage3.py \
  --features ./features/stage2_seed0_test.npz \
  --clusters 10 \
  --anchors 1000 \
  --anchor-neighbors 3 \
  --seed 0 \
  --output-json ./results/stage2_seed0_test.json
```

若只想快速检查 K-Means 和分类头流程，可添加 `--skip-anchor`。

## 代码结构

| 路径 | 作用 |
| --- | --- |
| `prepare_protocol.py` | 生成并验证分层的 1% 标签划分 |
| `data_protocol.py` | CIFAR-10 数据加载、增强和数据集包装 |
| `lora_vit.py` | 将 LoRA 注入 ViT attention 层 |
| `train_stage1_ssl.py` | Stage 1 自监督训练 |
| `train_stage2_semi.py` | Stage 2 半监督训练 |
| `checkpointing.py` | 检查点保存与 LoRA 状态加载 |
| `extract_stage3_features.py` | 使用 EMA/student 权重确定性提取特征 |
| `evaluate_stage3.py` | 分类、K-Means 与锚图聚类评估 |
| `anchor_spec.py` | Anchor Graph Spectral Clustering |
| `splits/` | 纳入版本控制的固定实验划分 |
| `results/` | 纳入版本控制的轻量结果 JSON |
| `newplan.md` | 当前三阶段实验设计及约束 |
| `train_lora_*.py`、`extract_features_any.py` | 早期实验入口，保留用于对照 |

`CIFAR-10/train.py` 与 `CIFAR-10/resnet.py` 是独立的 ResNet 基线代码，不属于上述三阶段主线。
`xiugai.md` 记录较早的修改方案，继续开发时应以当前源码和 `newplan.md` 为准。

## 继续开发建议

- 新实验使用新的 `seed` 和独立输出目录，避免覆盖 `seed0` 基线。
- 修改数据划分、数据增强或评估协议时，应同时更新结果 JSON 中的实验说明。
- 检查点、`.npz` 特征和原始数据不要直接提交到 Git；可通过 GitHub Release、对象存储或
  Git LFS 单独发布，并在 README 中记录下载地址和校验值。
- 对模型或损失函数的改动，至少运行语法检查、模块导入检查和小样本特征提取，再开始长时间训练。
- 正式结果应保存完整命令、随机种子、环境版本和硬件信息。

## 快速检查

无需重新训练即可执行：

```bash
python -m compileall -q .
python test_utils.py
python prepare_protocol.py --help
python train_stage1_ssl.py --help
python train_stage2_semi.py --help
python extract_stage3_features.py --help
python evaluate_stage3.py --help
```

## 大文件说明

本地目录中已有的 CIFAR 数据、`weights/`、`features/*.npz`、TensorBoard 日志和 Python
缓存由 `.gitignore` 排除，因此初始化仓库或执行 `git add .` 不会上传这些生成物。本项目代码和固定
实验元数据足以从头复现；如需直接复核现有模型结果，需要另行取得对应检查点。

