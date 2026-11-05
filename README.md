# 计算机视觉实验四：MLP 多层感知机——MNIST 手写数字识别


## 一、实验目的

理解多层感知机（MLP）的基本结构与工作原理，掌握利用 PyTorch 构建全连接神经网络并完成 MNIST 手写数字分类的方法。通过比较基础模型与改进模型，分析网络层数、激活函数、归一化、Dropout 和优化器等因素对训练收敛及分类性能的影响。

## 二、实验内容与环境

- Python 3.13
- PyTorch、torchvision、NumPy、Matplotlib、scikit-learn
- 开发环境：PyCharm
- 数据集：MNIST

MNIST 训练集包含 60,000 张 28×28 灰度图像，测试集包含 10,000 张 28×28 灰度图像。输入图像展平为 784 维向量，输出为 0～9 共 10 个类别。本次设置 batch_size=2048，训练轮数 epochs=10。

## 三、实验原理与模型结构

### 1. 基础 MLP

网络结构为 `784 → 128 → 10`。隐藏层使用 ReLU 激活函数，采用交叉熵损失函数和 SGD 优化器（学习率 0.1）训练。

### 2. 改进 MLP

网络结构为 `784 → 256 → 128 → 64 → 10`。隐藏层加入 BatchNorm1d 和 GELU，并使用 Dropout(0.3)降低过拟合风险；优化器采用 Adam（学习率 0.001）。

## 四、实验步骤

1. 使用 torchvision.datasets.MNIST 加载训练集和测试集，并通过 ToTensor 转换图像。
2. 使用 DataLoader 按批次读取数据，batch_size 设置为 2048。
3. 构建基础 MLP，使用 ReLU、交叉熵损失和 SGD 完成 10 轮训练。
4. 构建改进 MLP，加入 BatchNorm、GELU 和 Dropout，使用 Adam 完成 10 轮训练。
5. 在相同测试集上计算损失和准确率，生成损失曲线、准确率对比图及混淆矩阵。
6. 保存结果图片和模型权重。

本次运行确认训练数据形状为 `[60000, 28, 28]`，测试数据形状为 `[10000, 28, 28]`，程序正常结束，退出代码为 0。

## 五、实验结果与分析

### 1. 基础 MLP 训练损失

| Epoch | Training Loss |
|---:|---:|
| 1 | 1.9499 |
| 2 | 1.1114 |
| 3 | 0.7210 |
| 4 | 0.5719 |
| 5 | 0.4963 |
| 6 | 0.4508 |
| 7 | 0.4202 |
| 8 | 0.3980 |
| 9 | 0.3813 |
| 10 | 0.3678 |

基础模型测试集 Loss 为 **0.3453**，准确率为 **90.30%**。训练损失逐步下降，说明模型能够学习 MNIST 图像中的数字特征，训练过程整体收敛。

**对应 Python 代码（基础模型训练与损失曲线绘制）：**

```python
# 创建并训练基础 MLP
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
base_losses = train_model(model, optimizer, "Original MLP")
base_loss, base_acc, base_cm = evaluate_model(model, "Original MLP")

# 绘制并保存基础模型训练损失曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(base_losses) + 1), base_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Original MLP Training Loss")
plt.xticks(range(1, len(base_losses) + 1))
plt.grid(True)
plt.tight_layout()
plt.savefig("results/original_loss.png", dpi=220)
plt.close()
```

![基础 MLP 训练损失曲线](results/original_loss.png)

*图 1 基础 MLP 模型实际训练损失曲线*

### 2. 改进 MLP 训练损失

| Epoch | Training Loss |
|---:|---:|
| 1 | 1.1326 |
| 2 | 0.4764 |
| 3 | 0.2906 |
| 4 | 0.2073 |
| 5 | 0.1611 |
| 6 | 0.1330 |
| 7 | 0.1144 |
| 8 | 0.0976 |
| 9 | 0.0864 |
| 10 | 0.0765 |

改进模型测试集 Loss 为 **0.0688**，准确率为 **97.96%**。训练损失下降明显，测试集损失较低，表明该模型在本次实验中取得了更好的分类表现。

**对应 Python 代码（改进模型训练与损失曲线绘制）：**

```python
# 创建并训练改进 MLP
improved_model = ImprovedMLP().to(device)
improved_optimizer = optim.Adam(improved_model.parameters(), lr=1e-3)
improved_losses = train_model(
    improved_model, improved_optimizer, "Improved MLP"
)
improved_loss, improved_acc, improved_cm = evaluate_model(
    improved_model, "Improved MLP"
)

# 绘制并保存改进模型训练损失曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(improved_losses) + 1), improved_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Improved MLP Training Loss")
plt.xticks(range(1, len(improved_losses) + 1))
plt.grid(True)
plt.tight_layout()
plt.savefig("results/improved_loss.png", dpi=220)
plt.close()
```

![改进 MLP 训练损失曲线](results/improved_loss.png)

*图 2 改进 MLP 模型实际训练损失曲线*

### 3. 测试准确率对比

| 模型 | 测试 Loss | 测试准确率 | 训练轮数 |
|---|---:|---:|---:|
| 基础 MLP | 0.3453 | 90.30% | 10 |
| 改进 MLP | 0.0688 | 97.96% | 10 |

改进模型准确率提高 **7.66 个百分点**，测试损失由 0.3453 降至 0.0688，减少 0.2765。

**对应 Python 代码（计算并绘制测试准确率对比图）：**

```python
# base_acc 和 improved_acc 由 evaluate_model() 返回
names = ["Original MLP", "Improved MLP"]
values = [base_acc * 100, improved_acc * 100]

plt.figure(figsize=(7, 5))
bars = plt.bar(names, values)
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.ylim(0, 100)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1,
             f"{value:.2f}%", ha="center")
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=220)
plt.close()
```

![两种模型测试准确率对比](results/accuracy_comparison.png)

*图 3 两种模型实际测试准确率对比*

### 4. 混淆矩阵分析

基础模型混淆矩阵的主要计数集中在主对角线上，说明多数数字样本被正确分类；但主对角线以外仍存在一定数量的误分类，部分外形相近的数字容易混淆。

**对应 Python 代码（统计基础模型预测结果并绘制混淆矩阵）：**

```python
# 在 evaluate_model() 中逐批预测并累计真实标签与预测标签
confusion = np.zeros((10, 10), dtype=np.int64)
with torch.no_grad():
    for x, y in test_loader:
        x = x.view(x.shape[0], -1).to(device)
        y = y.to(device)
        output = model(x)
        pred = output.argmax(dim=1)
        for true_label, pred_label in zip(
            y.cpu().numpy(), pred.cpu().numpy()
        ):
            confusion[true_label, pred_label] += 1
base_cm = confusion

# 绘制并保存基础模型混淆矩阵
plt.figure(figsize=(8, 7))
plt.imshow(base_cm, cmap="Blues")
plt.title("Original MLP Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(base_cm[i, j]),
                 ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig("results/original_confusion_matrix.png", dpi=220)
plt.close()
```

![基础 MLP 混淆矩阵](results/original_confusion_matrix.png)

*图 4 基础 MLP 模型实际测试集混淆矩阵*

改进模型混淆矩阵的主对角线更加集中，非对角线误分类数量整体减少，表明模型对各数字类别的区分能力更强。混淆矩阵由模型对 10,000 张测试图像的预测结果统计得到。

**对应 Python 代码（绘制改进模型混淆矩阵）：**

```python
# improved_cm 由 evaluate_model(improved_model, ...) 返回
plt.figure(figsize=(8, 7))
plt.imshow(improved_cm, cmap="Blues")
plt.title("Improved MLP Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(improved_cm[i, j]),
                 ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig("results/improved_confusion_matrix.png", dpi=220)
plt.close()
```

![改进 MLP 混淆矩阵](results/improved_confusion_matrix.png)

*图 5 改进 MLP 模型实际测试集混淆矩阵*

## 六、实验结论

本次实验成功完成 MNIST 数据集上的基础 MLP 与改进 MLP 训练和测试。基础模型测试准确率为 90.30%，改进模型为 97.96%，提升 7.66 个百分点；改进模型测试损失由 0.3453 降低至 0.0688。实验结果说明，在本次实验设置下，增加网络层次并结合 BatchNorm、GELU、Dropout 和 Adam 优化器，能够有效提升 MNIST 分类性能。

## 七、实验总结与心得

通过本次实验，我掌握了使用 PyTorch 构建多层感知机、加载 MNIST 数据集、设置损失函数与优化器、执行模型训练以及评估分类效果的基本流程。基础模型能够实现手写数字识别，但在部分数字类别上仍存在混淆。改进模型通过更深的网络结构和多种训练策略，使损失下降更快、测试准确率更高。通过训练损失曲线和混淆矩阵，可以从训练过程与分类结果两个角度分析模型性能。实验也让我认识到，模型效果需要以实际运行数据和测试结果为依据，不能仅凭网络结构推断。

## 八、文件说明

- `README.md`：实验报告文本及结果图片展示。
- `experiment4_MLP.py`：完整 Python 实验代码。
- `results/`：五张真实实验结果图片。

