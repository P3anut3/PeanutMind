# PeanutMind 手撕导学

> 当前结论：完成 `trainer/train_pretrain.py` 后，不要立刻进入 DPO/PPO/GRPO。先补一个最小推理与验收闭环，再实现 `SFTDataset`，最后复用现有训练骨架完成 `train_full_sft.py`。

## 1. 前置知识（面试高频标注）

| 知识点 | 为何需要 | 在本项目中的位置 | 高频度 |
| --- | --- | --- | --- |
| 自回归语言模型与标签右移 | 理解模型为何用第 `t` 位预测第 `t+1` 位 | `model/model.py` 的 CausalLM loss | ★★★★★ |
| Chat Template 与 loss mask | SFT 的核心不是换训练循环，而是只监督 assistant 回答 | 待实现于 `dataset/lm_dataset.py` | ★★★★★ |
| KV Cache | 验证训练出的权重是否能高效逐 token 生成 | `model/model.py` 的 `generate` 与 attention cache | ★★★★★ |
| 梯度累积、AMP、梯度裁剪 | 理解有效 batch size、数值稳定性与显存取舍 | `trainer/train_pretrain.py` | ★★★★☆ |
| DDP 与分布式采样 | 避免多卡重复读样本，理解同步梯度 | `trainer/train_pretrain.py`、`trainer/trainer_utils.py` | ★★★★☆ |
| checkpoint 与可复现性 | 区分“模型权重”与“可恢复训练状态” | `trainer/trainer_utils.py` | ★★★★☆ |
| PPL 与 held-out loss | 不能只凭训练 loss 判断是否学到或过拟合 | 当前仓库待补 | ★★★★☆ |

## 2. 重点亮点与学习顺序（先看这个）

| 亮点标题 | 为什么重要 | 通用技术关键词 | 先看哪些文件 | 建议学习顺序 |
| --- | --- | --- | --- | --- |
| 推理验收闭环 | 先证明 checkpoint、tokenizer、生成链路相互兼容 | greedy decoding、sampling、KV Cache | `model/model.py`、待新增 `eval_llm.py` | 1 |
| 监督边界建模 | SFT 最关键的新知识是“哪些 token 算 loss” | chat template、assistant-only mask、`-100` | `dataset/lm_dataset.py` | 2 |
| 训练阶段迁移 | 预训练和 SFT 的骨架几乎相同，重点是初始化权重和超参变化 | transfer learning、低学习率、全参数微调 | `trainer/train_pretrain.py`、待新增 `trainer/train_full_sft.py` | 3 |
| 可恢复训练 | 训练中断后需恢复模型、优化器、scaler 和进度 | atomic checkpoint、resume、world size | `trainer/trainer_utils.py` | 4 |
| 定量评估 | 建立“能生成”与“生成得好”的区别 | held-out loss、PPL、固定 prompts、回归测试 | 当前仓库待补 | 5 |

## 3. 必备知识点

- [x] 能解释 causal mask、RoPE、GQA、RMSNorm、SwiGLU。
- [x] 能解释预训练输入、标签右移及 padding 的 `-100`。
- [x] 能解释梯度累积、AMP、裁剪、余弦学习率和 DDP。
- [ ] 能手算一条两轮对话经过 chat template 后，哪些 token 的 label 是 `-100`。
- [ ] 能分别用 `use_cache=False/True` 生成，并验证输出前缀一致。
- [ ] 能从预训练 checkpoint 初始化 SFT，而不是错误地随机初始化。
- [ ] 能用验证集 loss/PPL 和固定 prompt 对比 pretrain 与 SFT。

## 4. 推荐阅读（结合仓库）

| 主题 | 通用技术点 | 建议阅读位置 | 预计时间 | 读完能回答什么 |
| --- | --- | --- | --- | --- |
| 当前训练闭环 | optimizer step、梯度累积、保存恢复 | `trainer/train_pretrain.py` | 45 分钟 | 一个 batch 如何变成一次参数更新？ |
| 权重与训练状态 | state dict、resume checkpoint | `trainer/trainer_utils.py` | 30 分钟 | 为什么只保存 `.pth` 权重不能无损续训？ |
| 自回归输出契约 | logits、label shift、cache | `model/model.py` | 60 分钟 | 为何 `logits[:-1]` 对齐 `labels[1:]`？ |
| 预训练数据 | BOS/EOS、padding、ignore index | `dataset/lm_dataset.py` | 20 分钟 | padding 为什么不能参与 loss？ |
| 官方 SFT 数据逻辑 | chat template、assistant-only loss | [MiniMind 官方 `lm_dataset.py`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py) | 45 分钟 | 如何只训练 assistant 回答且支持多轮对话？ |
| 官方 SFT 训练入口 | 阶段权重迁移与超参 | [MiniMind 官方 `train_full_sft.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py) | 30 分钟 | SFT 与 pretrain 的代码差异究竟在哪里？ |

## 5. 自学提醒

若某文件或原理看不懂，请继续追问 AI；本技能负责给学习路径与题目，不提供逐行讲解。

## 6. 项目技术定位

**AI / 深度学习系统。** 依据是仓库从零实现了 decoder-only Transformer、预训练数据管线、原生 PyTorch 训练与自回归生成，并正在向指令微调阶段演进。

## 7. 核心原理解析

### 7.1 为什么先做推理 smoke test，再做 SFT

问题：训练 loss 下降只证明优化器能拟合 batch，不能证明保存的权重可加载、tokenizer 一致、EOS 正确或 KV Cache 可用。机制：固定随机种子与 prompt，分别以 greedy、无 cache、带 cache 三种方式生成短文本，并检查 checkpoint 加载和输出形状。在本项目中的落点：复用 `trainer/trainer_utils.py::init_model` 和 `model/model.py::generate`，新增一个很薄的 `eval_llm.py` 即可。

### 7.2 SFT 真正新增的是 loss mask

问题：若对 system、user、padding 全部计算 loss，模型会学习复述用户输入，而不是只学习回答。机制：先用 tokenizer 的 chat template 将多轮消息序列化，再定位每个 assistant 内容区间，仅保留这些 token 的 label，其余设为 `-100`。在本项目中的落点：为 `dataset/lm_dataset.py` 增加 `SFTDataset`，先拿一条两轮样本逐 token 打印并人工验算。

### 7.3 SFT 训练循环为何可以复用

问题：容易误以为新阶段需要重写训练框架。机制：pretrain 与 full SFT 都是 teacher forcing 下的 next-token cross entropy；变化主要在数据集、初始权重、学习率、序列长度和输出权重名。在本项目中的落点：从 `trainer/train_pretrain.py` 提炼或复制一个最小版本，默认 `from_weight=pretrain`、较低学习率，并替换为 `SFTDataset`。

### 7.4 checkpoint 有两种用途

问题：部署/推理只需模型权重，续训却还需要 optimizer、GradScaler、epoch、step 等状态。机制：轻量权重写到 `out/`，完整恢复状态写到 `checkpoints/`，临时文件完成后再原子替换。在本项目中的落点：`trainer/trainer_utils.py::lm_checkpoint` 已覆盖两种语义，后续 SFT 应复用而不是另造格式。

### 7.5 评估要区分 base model 与 chat model

问题：pretrain 模型目标是文本续写，SFT 模型目标是遵循对话指令，不能只用同一组聊天问题凭感觉比较。机制：pretrain 用 held-out NLL/PPL 与续写样例，SFT 用固定指令集检查格式、相关性和终止行为。在本项目中的落点：先保留 100～1000 条未训练文本作验证集，再增加 10 条固定中文 prompts；规模只是建议，结果目前均为待测。

## 8. 关键设计决策

| 决策 | 备选 | 取舍 | 风险 | 验证 |
| --- | --- | --- | --- | --- |
| 先 eval 后 SFT | 直接复制官方 SFT | 多写一个小脚本，但能尽早暴露权重/生成问题 | 把随机采样差异误判成 bug | 先用 greedy 和固定 seed |
| assistant-only loss | 全序列 loss | 更符合指令微调目标，但 mask 更易写错 | BOS/EOS 模式匹配错误造成全 `-100` | 逐 token 打印 label，断言有效 label 数大于 0 |
| 复用训练骨架 | 完全重写 trainer | 复用更聚焦阶段差异；重写有助练习但重复较多 | 两份脚本日后漂移 | 学习阶段先复制并明确 diff，之后再抽公共 trainer |
| full SFT | 先 LoRA | full SFT 更容易理解权重更新全貌；LoRA 更省显存 | 小显存 OOM | 先用小 hidden size / 小 batch 做 smoke test |
| 暂缓 RL | 直接上 DPO/GRPO | 先建立可靠 SFT 基线，便于定位后续增益 | 学习路线拉长 | SFT 固定集验收后再进入偏好学习 |

## 9. 量化与验证（待测，建议）

1. **训练正确性（立即做）**：构造 8～32 条小数据，确认 loss 能明显下降并可过拟合；这是管线测试，不代表泛化能力。
2. **checkpoint 一致性（立即做）**：保存前后对同一输入做 `eval + greedy`，比较 logits 或生成 token 是否一致。
3. **KV Cache 一致性（立即做）**：同一 prompt 使用/不使用 cache，greedy 模式下比较生成 token；允许极小浮点差异，不应发生系统性分叉。
4. **预训练基线（SFT 前）**：在 held-out 文本上记录 loss/PPL，保存 5 条续写结果，当前数值为待测。
5. **SFT mask（实现时）**：统计每条样本有效监督 token 数；断言 padding、user、system 均不参与 loss，assistant EOS 是否监督要明确。
6. **阶段对比（SFT 后）**：同一组 10～30 条固定 prompts 对比 pretrain 与 full SFT 的指令遵循、终止和格式，结果待测。

## 10. 你接下来实际应该写什么

建议拆成三个小提交：

1. `eval: add checkpoint generation smoke test`：加载 `out/pretrain_*.pth`，用 greedy 生成，验证 cache on/off。
2. `data: implement assistant-only SFTDataset`：实现 chat template、截断、padding 与 assistant label mask，附最小断言或调试打印。
3. `train: add full SFT stage`：从 pretrain 权重初始化，使用更低学习率和 `SFTDataset`；训练循环可复用，不必追求与官方逐行一致。

官方当前 README 把预训练和 full SFT 都列为基础阶段，并说明 mini 数据组合适合快速复现；本导学与官方阶段方向一致，但额外加入了推理验收与定量基线，以服务“手撕并真正理解”的目标：[MiniMind 官方训练说明](https://github.com/jingyaogong/minimind#-模型训练)。
