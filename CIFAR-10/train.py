import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import ResNet18
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10')
    parser.add_argument('--logdir', type=str, default='./runs', help='Directory to save TensorBoard logs')
    parser.add_argument('--epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for training and testing')
    return parser.parse_args()

# 动态创建日志目录
def create_log_dir(base_dir):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    log_dir = base_path / 'train'
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        i = 1
        while True:
            new_log_dir = base_path / f'train{i}'
            if not new_log_dir.exists():
                new_log_dir.mkdir(parents=True, exist_ok=True)
                log_dir = new_log_dir
                break
            i += 1
    
    return str(log_dir)

# 数据预处理
def get_data_loaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# 定义模型、损失函数和优化器
def setup_model(learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, device

# 训练函数
def train(epoch, trainloader, model, criterion, optimizer, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 10 == 9:  # 每10个batch打印一次
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 计算平均损失和准确率
    avg_loss = running_loss / len(trainloader)
    accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch}\tAverage loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)')
    writer.add_scalar('Train/Average Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)

# 测试函数
def test(epoch, testloader, model, criterion, writer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(testloader)
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')
    writer.add_scalar('Test/Average Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)

# 主程序入口点
if __name__ == '__main__':
    args = parse_args()
    log_dir = create_log_dir(args.logdir)
    writer = SummaryWriter(log_dir=log_dir)

    trainloader, testloader = get_data_loaders(args.batch)
    model, criterion, optimizer, device = setup_model(learning_rate=0.001)

    # 开始训练
    for epoch in range(1, args.epoch + 1):
        train(epoch, trainloader, model, criterion, optimizer, writer)
        test(epoch, testloader, model, criterion, writer)

    # 关闭writer
    writer.close()