目标：我需要增加一个对比学习，只使用1%的标注，在CIFAR-10上指标ACC至少0.981，NMI至少0.954，ARI至少0.959

# 步骤 1 修改 utils.py（按你给的代码为主，补充缺失函数）

1. 打开项目里的`utils.py`，拉到文件最末尾
2. 粘贴你写的完整 SimSiam 工具代码

```
# ========== 新增：SimSiam对比学习工具 ==========
import torch
import torch.nn as nn
import torch.nn.functional as F

def simsiam_aug(img_size=224):
    """
    SimSiam用的强增强：同一张图变两次，每次不一样
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

class TwoViewDataset(torch.utils.data.Dataset):
    """
    包装原有数据集，返回同一张图的两个增强视图
    """
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform
  
    def __len__(self):
        return len(self.base)
  
    def __getitem__(self, idx):
        img, label = self.base[idx]
        x1 = self.transform(img)  # 第一次增强
        x2 = self.transform(img)  # 第二次增强（随机性导致不同）
        return x1, x2, label

def simsiam_loss(p, z):
    """
    SimSiam核心损失：预测器输出p去逼近投影输出z（z停止梯度）
    """
    z = z.detach()  # 关键！z不更新
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=1).mean()

class SimSiamHead(nn.Module):
    """
    SimSiam的投影头+预测头
    """
    def __init__(self, dim=768, proj_dim=2048, pred_dim=512):
        super().__init__()
    
        # 投影头：768 → 2048 → 2048
        self.projector = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    
        # 预测头：2048 → 512 → 2048（SimSiam关键，防止坍塌）
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )
  
    def forward(self, feat):
        z = self.projector(feat)   # 投影特征
        p = self.predictor(z)       # 预测特征
        return p, z
```

3. 在上面这段代码**最下方追加 2 个配套函数**（解决特征同质化、双向对称损失，我补充的核心缓解项）

```
def total_sim_loss(z1, p1, z2, p2):
    """双向对称SimSiam总损失"""
    loss_1 = simsiam_loss(p1, z2)
    loss_2 = simsiam_loss(p2, z1)
    return (loss_1 + loss_2) / 2

def sep_reg(feat_batch):
    """类间分离正则，缓解多视图带来特征全部趋同问题"""
    b_size = feat_batch.shape[0]
    feat_norm = F.normalize(feat_batch, dim=-1)
    gram_matrix = torch.matmul(feat_norm, feat_norm.T)
    eye_mask = torch.eye(b_size, device=feat_batch.device)
    # 只计算不同样本间相似度
    cross_sim = gram_matrix * (1 - eye_mask)
    return cross_sim.sum() / (b_size * (b_size - 1))
```

4. 保存`utils.py`，执行导入测试验证是否成功

```
python
>>> from utils import simsiam_aug, TwoViewDataset, SimSiamHead, total_sim_loss, sep_reg
>>> print("导入全部成功，无报错即可进入下一步")
```

# 步骤 2 新建 Stage1 预训练脚本 train\_simsiam\_pretrain.py（无标签 SimSiam 预训练）

## 功能说明

只用全部无标注数据，训练 15 轮，仅更新 LoRA+SimSiam 头，不使用任何标签，输出预训练 LoRA 权重

```
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import loralib as lora
# 统一从utils导入所有SimSiam工具，无需重复定义
from utils import get_dataset, simsiam_aug, TwoViewDataset, SimSiamHead, total_sim_loss, sep_reg
from lora_vit import VITWithLoRA

# 固定全局随机种子，保证实验可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 固定超参数
DEVICE = "cuda"
BATCH_SIZE = 64
EPOCHS_STAGE1 = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-1
REG_COEFF = 0.1  # 分离正则权重

if __name__ == "__main__":
    # 1. 加载完整训练集，label_ratio=0代表全部视为无标注样本
    full_train, _ = get_dataset(dataset="cifar10", label_ratio=0)
    aug_func = simsiam_aug()
    # 使用你定义的TwoViewDataset包装数据集，输出x1,x2,label
    train_dataset = TwoViewDataset(full_train.dataset, aug_func)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. 初始化ViT-LoRA，主干完全冻结，仅LoRA可训练
    model = VITWithLoRA(r=16, alpha=32).to(DEVICE)
    lora.mark_only_lora_as_trainable(model)
    sim_head = SimSiamHead().to(DEVICE)

    # 3. 优化器：只更新LoRA参数 + SimSiam投影/预测头
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(sim_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    print("===== Stage1：无标签SimSiam预训练开始 =====")
    for epoch in range(EPOCHS_STAGE1):
        model.train()
        sim_head.train()
        epoch_loss = 0.0
        for (x1, x2, _) in tqdm(train_loader):
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            # 混合精度加速训练，降低显存占用
            with torch.cuda.amp.autocast():
                # 提取两张视图的CLS特征
                feat1 = model(x1)[:, 0, :]
                feat2 = model(x2)[:, 0, :]
                p1, z1 = sim_head(feat1)
                p2, z2 = sim_head(feat2)
                # 总损失 = SimSiam双向损失 + 类间分离正则
                loss_sim = total_sim_loss(z1, p1, z2, p2)
                loss_reg = sep_reg(feat1)
                total_loss = loss_sim + REG_COEFF * loss_reg
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} | 平均损失：{avg_loss:.4f}")
  
    # 只保存LoRA增量权重（loralib标准保存方式，体积小）
    save_path = "./weights/lora_pretrain_cifar10.pth"
    torch.save(lora.lora_state_dict(model), save_path)
    print(f"Stage1预训练完成，权重保存路径：{save_path}")
```

## 运行前操作

项目根目录手动新建`weights`文件夹存放所有模型权重

# 步骤 3 新建 Stage2 微调脚本 train\_lora\_semi.py（少量标签微调）

## 功能说明

加载 Stage1 预训练好的 LoRA 权重，冻结 SimSiam 头，只用少量标注样本训练，训练轮次与原论文保持 5 轮，完全复用原有分类训练逻辑

```
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import loralib as lora
from utils import get_dataset, SimSiamHead
from lora_vit import VITWithLoRA

# 固定全局随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda"
BATCH_SIZE = 64
EPOCHS_STAGE2 = 5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-1
NUM_CLS = 10  # CIFAR10类别数

if __name__ == "__main__":
    # 【修改此处切换标注比例 0.01 / 0.05 / 0.10】
    LABEL_RATIO = 0.01
    # 1. 加载分层均衡抽样的标注数据集（复用原有utils逻辑）
    train_labeled, _ = get_dataset(dataset="cifar10", label_ratio=LABEL_RATIO)
    train_loader = DataLoader(train_labeled, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. 加载Stage1预训练的LoRA权重
    model = VITWithLoRA(r=16, alpha=32).to(DEVICE)
    pretrain_weight_path = "./weights/lora_pretrain_cifar10.pth"
    lora_weight = torch.load(pretrain_weight_path)
    model.load_state_dict(lora_weight, strict=False)
    lora.mark_only_lora_as_trainable(model)

    # 3. 冻结SimSiam全部参数，不再参与训练（核心：消除损失冲突）
    sim_head = SimSiamHead().to(DEVICE)
    for param in sim_head.parameters():
        param.requires_grad = False

    # 临时分类头，和原训练代码保持一致
    cls_head = nn.Linear(768, NUM_CLS).to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss()

    # 优化器：仅更新LoRA + 临时分类头
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(cls_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"===== Stage2 微调 | 标注比例：{LABEL_RATIO*100}% =====")
    for epoch in range(EPOCHS_STAGE2):
        model.train()
        cls_head.train()
        total_ce = 0.0
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                feat = model(x)[:, 0, :]
                logits = cls_head(feat)
                loss = ce_loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_ce += loss.item()
        avg_ce = total_ce / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE2} | 交叉熵损失：{avg_ce:.4f}")
  
    # 保存本次微调后的LoRA权重
    save_name = f"./weights/lora_label_{int(LABEL_RATIO*100)}_cifar10.pth"
    torch.save(lora.lora_state_dict(model), save_name)
    print(f"微调完成，模型已保存：{save_name}")
```

# 步骤 4 完整实验运行顺序（严格不能颠倒）

## 4.1 前置依赖安装（终端执行）

```
pip install loralib umap-learn scipy matplotlib tqdm torch torchvision
```

## 4.2 第一步：Stage1 预训练（仅运行 1 次）

```
python train_simsiam_pretrain.py
```

等待 15 轮训练完成，weights 文件夹生成`lora_pretrain_cifar10.pth`

## 4.3 第二步：分三次运行 Stage2 微调

1. 修改代码`LABEL_RATIO = 0.01`，运行脚本，得到 1% 标注模型

```
python train_lora_semi.py
```

2. 修改`LABEL_RATIO = 0.05`，再次运行，得到 5% 标注模型
3. 修改`LABEL_RATIO = 0.10`，再次运行，得到 10% 标注模型

## 4.4 第三步：原始基线对照（消融实验）

运行你原本未添加 SimSiam 的训练代码，获取 10% 标注下的原始聚类分数

```
python train_lora.py --label_ratio 0.1
```

## 4.5 第四步：提取特征 + 聚类（原有代码完全不用修改）

1. 修改`extract_features_any.py`，分别加载四组模型（基线、1%、5%、10%）提取全部数据集特征
2. 运行`anchor_spec.py`，自动输出 ACC、NMI、ARI 聚类指标
3. 使用`vis_2d.py`绘制特征 UMAP 可视化、标签比例折线图
