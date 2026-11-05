# -*- coding: utf-8 -*-
"""
实验四：MLP 多层感知机——MNIST 手写数字识别
依据 GitHub README 的实验设置整理：
基础模型：784-128-10，ReLU，SGD(lr=0.1)
改进模型：784-256-128-64-10，GELU、BatchNorm、Dropout(0.3)，Adam(lr=1e-3)
batch_size=2048，epochs=10
运行后自动保存损失曲线、混淆矩阵和准确率对比图。
"""
 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
 
# 1. 基本设置
torch.manual_seed(42)
np.random.seed(42)
batch_size = 2048
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("results", exist_ok=True)
print("当前设备：", device)
 
# 2. 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    'data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    'data', train=False, download=True, transform=transform
)
 
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)
print("训练数据形状：", train_dataset.data.shape)
print("测试数据形状：", test_dataset.data.shape)
 
# 3. 基础 MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
 
    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        return a2
 
# 4. 改进 MLP
class ImprovedMLP(nn.Module):
    def __init__(self):
        super(ImprovedMLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.3)
        self.act = nn.GELU()
 
    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
 
# 5. 通用训练函数
def train_model(model, optimizer, model_name):
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_num = 0
 
        for x, y in train_loader:
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
 
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item() * x.size(0)
            total_num += x.size(0)
 
        epoch_loss = total_loss / total_num
        history.append(epoch_loss)
        print(f"{model_name} Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")
    return history
 
# 6. 测试函数，返回混淆矩阵
def evaluate_model(model, model_name):
    model.eval()
    correct = 0
    total_loss = 0.0
    total_num = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
 
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
            output = model(x)
            loss = F.cross_entropy(output, y)
 
            pred = output.argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            total_num += x.size(0)
            correct += (pred == y).sum().item()
 
            y_np = y.cpu().numpy()
            pred_np = pred.cpu().numpy()
            for t, p in zip(y_np, pred_np):
                confusion[t, p] += 1
 
    avg_loss = total_loss / total_num
    accuracy = correct / total_num
    print(f"{model_name} 测试结果 - loss:{avg_loss:.4f}, acc:{accuracy:.4f}")
    return avg_loss, accuracy, confusion
 
# 7. 绘图函数
def save_loss_plot(losses, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.xticks(range(1, len(losses)+1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
 
def save_confusion_matrix(matrix, title, filename):
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(matrix[i, j]),
                     ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
 
# 8. 训练、测试基础模型
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
print(model)
base_losses = train_model(model, optimizer, "Original MLP")
base_loss, base_acc, base_cm = evaluate_model(model, "Original MLP")
save_loss_plot(base_losses, "Original MLP Training Loss",
               "results/original_loss.png")
save_confusion_matrix(base_cm, "Original MLP Confusion Matrix",
                      "results/original_confusion_matrix.png")
torch.save(model.state_dict(), "results/original_mlp.pth")
 
# 9. 训练、测试改进模型
improved_model = ImprovedMLP().to(device)
improved_optimizer = optim.Adam(improved_model.parameters(), lr=1e-3)
print(improved_model)
improved_losses = train_model(
    improved_model, improved_optimizer, "Improved MLP"
)
improved_loss, improved_acc, improved_cm = evaluate_model(
    improved_model, "Improved MLP"
)
save_loss_plot(improved_losses, "Improved MLP Training Loss",
               "results/improved_loss.png")
save_confusion_matrix(improved_cm, "Improved MLP Confusion Matrix",
                      "results/improved_confusion_matrix.png")
torch.save(improved_model.state_dict(), "results/improved_mlp.pth")
 
# 10. 准确率对比图
plt.figure(figsize=(7, 5))
names = ["Original MLP", "Improved MLP"]
values = [base_acc * 100, improved_acc * 100]
bars = plt.bar(names, values)
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.ylim(0, 100)
for bar, value in zip(bars, values):
    plt.text(bar.get_x()+bar.get_width()/2, value+1,
             f"{value:.2f}%", ha="center")
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=220)
plt.close()
 
print("\\n实验完成，图片及模型权重保存在 results 文件夹。")
