# PeanutMind 🥜

从零复现 [MiniMind](https://github.com/jingyaogong/minimind) 的小型大语言模型，用于学习 LLM 的训练全流程：**预训练 → SFT（指令微调）**，并附带命令行对话与网页聊天界面。

模型约 **63.9M 参数**，可以在单张消费级 GPU 上完成全部训练。

## 模型结构

| 配置项 | 值 |
|---|---|
| hidden_size | 768 |
| num_hidden_layers | 8 |
| vocab_size | 6400 |
| num_attention_heads | 8（GQA：4 个 KV head，head_dim=96） |
| intermediate_size | 2432（SwiGLU） |
| max_position_embeddings | 32768 |
| 总参数量 | 63.9M |

架构要点：RMSNorm 归一化、RoPE 旋转位置编码（支持 YaRN 外推）、GQA 分组查询注意力、SwiGLU 前馈网络、Pre-Norm Transformer Block，以及可选的 MoE 结构（`use_moe=1` 时启用，4 个专家 + Top-1 路由 + 负载均衡辅助损失）。

## 目录结构

```
PeanutMind/
├── model/
│   ├── model.py          # 模型定义（PeanutMindConfig / PeanutMindForCausalLM）
│   ├── model_lora.py     # LoRA 低秩微调
│   ├── tokenizer.json    # 分词器（vocab 6400）
│   └── tokenizer_config.json
├── dataset/
│   ├── lm_dataset.py     # PretrainDataset / SFTDataset
│   ├── pretrain_t2t_mini.jsonl   # 预训练数据（约 1.2GB）
│   └── sft_t2t_mini.jsonl        # SFT 数据（约 1.6GB）
├── trainer/
│   ├── train_pretrain.py # 预训练脚本
│   ├── train_sft.py      # 指令微调脚本
│   └── trainer_utils.py  # 模型加载 / 日志 / checkpoint 等工具
├── out/                  # 训练产物（权重）
├── eval_llm.py           # 命令行对话 / 评测
├── web_demo.py           # Streamlit 网页聊天界面
└── requirements.txt
```

## 快速开始

### 1. 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载数据

数据来自 ModelScope 的 [gongjy/minimind_dataset](https://modelscope.cn/datasets/gongjy/minimind_dataset)，按需单独下载（mini 系列即可）：

```bash
modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset pretrain_t2t_mini.jsonl
modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset sft_t2t_mini.jsonl
```

### 3. 训练

**预训练**（从零开始，产出 `out/pretrain_768.pth`）：

```bash
python trainer/train_pretrain.py --device cuda:3
```

**SFT 指令微调**（基于预训练权重，产出 `out/full_sft_768.pth`）：

```bash
python trainer/train_sft.py --device cuda:3
```

中断后可从断点续训（恢复模型 + 优化器 + 学习率状态）：

```bash
python trainer/train_sft.py --device cuda:3 --from_resume 1
```

> 注意：续训需保持 `--seed`、`--batch_size`、`--max_seq_len`、`--data_path` 与上次一致，否则跳过的 batch 会对不上。

### 4. 评测 / 对话

命令行对话（`[0]` 自动跑内置题目，`[1]` 手动输入）：

```bash
python eval_llm.py --weight full_sft --device cuda:3
```

网页聊天界面：

```bash
streamlit run web_demo.py
```

浏览器打开 `http://localhost:8501`；远程服务器可用 SSH 隧道：

```bash
ssh -L 8501:localhost:8501 <user>@<server>
```

## 训练进度

- [x] 预训练（`pretrain_t2t_mini.jsonl`，2 epochs）
- [x] SFT 指令微调（`sft_t2t_mini.jsonl`，2 epochs）
- [ ] LoRA 微调（`model/model_lora.py` 已实现，训练脚本待补）
- [ ] DPO / RLHF（参考官方后续流程）
