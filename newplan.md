
审阅结论：`xiugai.md` 不建议直接执行。它的总体思路可以保留，但代码目前无法接入项目，而且“15 轮 SimSiam + 5 轮 1% CE”不足以支撑 `ACC≥0.981、NMI≥0.954、ARI≥0.959` 这样的目标。

**主要问题**

1. 方案代码会直接报错。

- [xiugai.md](D:/Project/Project/VScode/youyong-main/xiugai.md:130) 从 `utils` 导入 `get_dataset`，但 [utils.py](D:/Project/Project/VScode/youyong-main/utils.py:9) 没有该函数。
- 写成了 `VITWithLoRA`，项目实际类名是 [ViTWithLoRA](D:/Project/Project/VScode/youyong-main/lora_vit.py:29)。
- 构造参数应为 `lora_alpha=32`，不是 `alpha=32`。
- `ViTWithLoRA.forward()` 返回 `[B, 768]`，方案却使用 `model(x)[:, 0, :]`，会维度越界。见 [lora_vit.py](D:/Project/Project/VScode/youyong-main/lora_vit.py:48)。
- `full_train.dataset` 的假设不成立，普通 CIFAR10 Dataset 没有这个包装层。
- Stage2 创建并冻结一个全新的随机 `SimSiamHead`，但后续从未使用，没有任何效果。

2. `sep_reg` 的理论解释不成立。

它不是“类间分离正则”，因为无监督阶段根本不知道类别。它会把同类样本也相互推开，而且直接平均有正有负的余弦相似度，容易相互抵消。建议删除，改用 VICReg 方差/协方差正则，或者直接采用标准 SimSiam/MoCo v3。

3. 训练量明显不足。

1% CIFAR-10 只有 500 张标注图。Stage2 训练 5 轮、batch 64，实际只有约 40 次参数更新。这样的训练量很难稳定学习到能达到 98.1% 聚类 ACC 的语义结构。

4. 保存和加载协议不一致。

方案只保存 `lora_state_dict`，但 [extract_features_any.py](D:/Project/Project/VScode/youyong-main/extract_features_any.py:43) 按完整模型检查 missing keys，可能直接拒绝加载。分类头也没有保存，无法复现训练状态。

另外，`extract_features_any.py` 导入了当前项目不存在的 `vit_frozen.py`，所以“特征提取代码完全不用修改”也是错误的。

5. 指标口径必须先固定。

当前 [evaluate.py](D:/Project/Project/Project/VScode/youyong-main/evaluate.py:9) 中的 ACC 是经过匈牙利匹配的聚类 ACC，不是分类准确率。目标必须明确为：

- CIFAR-10 test 10,000 张图片；
- 使用 backbone embedding；
- Anchor Spectral 或 KMeans 聚类；
- test 标签只在最后计算 ACC/NMI/ARI 时使用；
- 不允许根据 test 指标选择超参数或 checkpoint。

6. “仅 1% 标注”的定义存在隐患。

当前 `pretrained=True` 使用的是带外部预训练权重的 ViT，通常包含 ImageNet 监督信息。若“1%”仅指 CIFAR-10 标签，这可以接受；若指整个流程只能使用 500 个标签，则当前预训练模型不满足要求。

**推荐方案**

我建议采用三阶段方案，而不是原文的简单两阶段。

### 阶段 0：统一实验协议

- 从 CIFAR-10 train 中固定抽取 500 张标注数据，每类严格 50 张。
- 其余 49,500 张作为无标签集；500 张也可以隐藏标签后参与无监督训练。
- 固定 `seed=0,1,2,3,4` 五套划分。
- test 集只做最终一次评估。
- checkpoint 统一保存：
  - `backbone_state`
  - `projector_state`
  - `classifier_state`
  - `ema_state`
  - 模型名、LoRA 配置、数据划分和训练参数。

### 阶段 1：全量无标签对比适配

使用全部 50,000 张训练图，不读取标签。

优先采用适合 ViT 的 MoCo v3/SimSiam 风格训练：

- 保留 ImageNet 预训练 ViT-B/16。
- LoRA 只注入最后 4～6 个 Transformer block 的 attention `qkv/proj`，不要替换所有 `Linear`。
- LoRA 初始建议 `r=16, alpha=32`。
- projector：`768 → 2048 → 256`。
- predictor：`256 → 512 → 256`。
- 训练约 100～200 epochs，而不是 15 epochs。
- AdamW、cosine decay、5～10 epoch warmup。
- CIFAR 增强使用 `RandomResizedCrop(224, scale=(0.5, 1.0))`、水平翻转、ColorJitter、灰度和轻量 blur。
- 删除 `sep_reg`；需要防坍塌时加入 VICReg variance/covariance 项。
- 每轮记录特征标准差和有效秩，用来检测表示坍塌。

现有 [train_lora_unsup.py](D:/Project/Project/VScode/youyong-main/train_lora_unsup.py:172) 已有 InfoNCE 雏形，建议重构它，而不是再创建一套接口不兼容的脚本。

### 阶段 2：1% 标签与无标签联合训练

这是达到目标的关键。不要只在 500 张图片上做 CE。

每个 batch 同时包含：

- 类别均衡的 labeled batch；
- 约 4～7 倍大小的 unlabeled batch；
- labeled 数据使用有放回采样，保证每轮有足够更新次数。

总损失建议：

```text
L = L_ce
  + 0.2 * L_supcon
  + lambda_u * L_fixmatch
```

其中：

- `L_ce`：500 张标注数据的分类交叉熵。
- `L_supcon`：监督对比损失，同类为正样本，直接建立类别紧凑结构。
- `L_fixmatch`：EMA teacher 对无标签弱增强产生伪标签，student 在强增强上学习。
- 伪标签阈值初始 `0.95`，配合类别分布校准，避免模型只预测少数类别。
- `lambda_u` 在前 10～20 epochs 从 0 增长到 1。
- 训练 100～300 epochs，并保存 EMA backbone。
- 分类头和 projector 都参与训练，但最终聚类优先使用归一化 backbone embedding。

这比“Stage1 SimSiam 后只做 5 轮 CE”更有希望，因为无标签数据在第二阶段仍持续参与优化。

### 阶段 3：无泄漏特征与聚类评估

修改 [extract_features_any.py](D:/Project/Project/VScode/youyong-main/extract_features_any.py:69)：

- 删除或补齐不存在的 `FrozenViT`。
- 严格校验 LoRA 参数是否真正加载。
- 默认加载 EMA backbone。
- 对原图和水平翻转图的特征取平均，再做 L2 normalize。
- 同时保存 train 和 test 特征，但聚类结果只在 test 上评估。

修改 [evaluate.py](D:/Project/Project/VScode/youyong-main/evaluate.py:19)：

- 固定报告 KMeans 和 Anchor Spectral 两套结果。
- KMeans 使用 `n_init=50`。
- Anchor 的 `m/s/sigma` 只能根据 train 特征稳定性选择，不能查看 test 标签后调参。
- 输出每个 seed 的结果、均值和标准差。
- 明确区分 `classification_acc` 与 `clustering_acc`。

**建议的消融顺序**

后续真正运行实验时，按以下顺序判断提升来自哪里：

1. 冻结预训练 ViT + KMeans。
2. 当前 1% CE-LoRA。
3. 无标签对比适配。
4. 对比适配 + 1% CE。
5. 对比适配 + CE + SupCon。
6. 对比适配 + CE + SupCon + FixMatch/EMA。
7. KMeans 与 Anchor Spectral 对比。

最终应以五个种子的平均值达到阈值为目标，而不是挑选某个最好 seed。以当前项目基础，原 `xiugai.md` 无法可靠达到目标；“对比预训练 + SupCon 语义约束 + 无标签伪标签联合训练 + EMA”是更合理、成功概率更高的实施路线。本次只完成了方案审阅，没有运行实验，也没有继续修改代码。
