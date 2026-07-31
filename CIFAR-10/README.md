## 可视化

```shell
# 可视化界面
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动可视化界面
tensorboard --logdir=/home/runs --port=37777 --bind_all
```



## 模型训练

```shell
# 安装git扩展仓库
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

# 安装扩展
apt-get install git-lfs
```

```shell
# 拉取数据集
git clone https://www.modelscope.cn/datasets/EFate1006/CIFAR-10.git
```

```shell
# cd
cd CIFAR-10

# 训练
python train.py --epoch=20 --batch=16 --logdir=/home/runs
```

