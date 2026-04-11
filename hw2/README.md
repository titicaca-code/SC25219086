# Homework 2 - CNN for SVHN Image Classification

## 1. 作业简介
本次作业基于 SVHN（Street View House Numbers）Format 2 数据集，使用卷积神经网络（CNN）完成 0~9 共 10 类数字图像分类任务。

本项目使用 PyTorch 实现了：
- 基础版 CNN（SimpleCNN）
- 增强版 CNN（EnhancedCNN）

并完成了以下内容：
- 模型训练与测试
- 训练损失与测试损失曲线绘制
- 训练准确率与测试准确率曲线绘制
- 最优模型保存
- 测试集混淆矩阵分析
- 错分样本可视化

---

## 2. 项目结构

```text
hw2
├─ README.md
├─ train.py
├─ model.py
├─ utils.py
├─ analyze.py
├─ requirements.txt
├─ data
│  └─ SVHN
├─ checkpoints
│  └─ best_enhanced.pth
└─ figures
   ├─ loss_curve.png
   ├─ accuracy_curve.png
   ├─ confusion_matrix.png
   └─ wrong_samples.png