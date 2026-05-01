# Homework 3 - RNN实现古诗生成

## 1. 作业简介
本次作业基于宋诗数据集，使用字符级 LSTM 模型实现固定格式古诗生成任务。  
实验中以“明月”为总起词，生成七言绝句，并通过训练 loss 曲线展示模型收敛情况。

---

## 2. 项目结构

```text
hw3
├─ data
│  ├─ poet.song.40000.json
│  ├─ poet.song.41000.json
│  ├─ poet.song.42000.json
│  ├─ poet.song.43000.json
│  ├─ qijue.txt
│  ├─ vocab.json
│  ├─ char2idx.json
│  └─ idx2char.json
├─ checkpoints
│  └─ best_poetry_lstm.pth
├─ figures
│  ├─ loss_curve.png
│  └─ loss_history.json
├─ outputs
│  └─ generated_poems.json
├─ preprocess.py
├─ train.py
├─ generate.py
├─ model.py
├─ utils.py
├─ plot_loss.py
├─ README.md
└─ requirements.txt