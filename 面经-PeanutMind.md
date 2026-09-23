# PeanutMind 项目面经

## 1. 项目简介（简历可用）

个人学习型复刻项目：基于 PyTorch 从零实现轻量 decoder-only 大语言模型及预训练链路，覆盖 GQA、RoPE、RMSNorm、SwiGLU、可选 MoE、自回归生成、混合精度、梯度累积、DDP 与断点续训；当前已完成模型和预训练阶段，SFT 与系统评测仍在推进。

## 2. 简历 bullet

- **训练链路构建：** 针对只理解模型结构、缺少端到端训练经验的问题，独立复刻从 JSONL 文本、分词与标签构造到前向、反向、优化器更新和权重落盘的预训练链路；通过统一输入输出契约串起数据、模型与训练模块，当前已形成可运行的预训练闭环，模型效果指标待基于验证集测量。
- **显存与数值稳定性治理：** 面向小算力训练约束，在训练环节加入混合精度、梯度累积与梯度裁剪，并按 step 调整余弦学习率；明确 fp16 需要动态缩放、bf16 通常不需要的边界，使有效 batch size 与单步显存占用可独立调节，吞吐与显存收益待实测。
- **分布式训练：** 为支持单机多卡扩展，引入进程组、设备绑定、分布式采样与梯度同步，并处理主进程日志和存盘边界；保证不同进程消费不重叠的数据分片，跨卡一致性和加速比仍需通过单卡/多卡对照验证。
- **训练状态恢复：** 区分推理权重与完整训练状态，保存模型、优化器、混合精度缩放器、epoch、step 及实验标识，并采用临时文件后原子替换的方式降低中断写坏风险；已具备跨进程数调整 step 的恢复逻辑，恢复后数值连续性待增加自动化测试。
- **推理性能机制：** 在自回归生成中实现 KV Cache、greedy/temperature/top-k/top-p 采样和重复惩罚，使训练模型具备独立生成能力；当前代码层面已打通缓存读写与位置偏移，cache 开关下的输出一致性和速度收益待测。
- **稀疏计算探索：** 在标准前馈层之外实现可选的 token 级专家路由、top-k 选择与负载均衡辅助损失，并将辅助项接入总训练目标；形成 dense/MoE 共用训练入口，专家利用率与质量收益尚未形成实验结论。

## 3. 面试问题

### 主题一：总体链路与训练目标

#### 1. 你为什么要手撕 MiniMind，当前完成到什么程度？

**第一人称口播：** 我做这个项目的目的不是再调用一次现成 Trainer，而是把一个 decoder-only 语言模型从数据到生成的关键接口亲手连起来。我先完成了 tokenizer 兼容、Transformer 主体、自回归 loss、预训练数据集和原生 PyTorch 训练脚本，又补了 AMP、梯度累积、裁剪、DDP 和断点续训。现在可以明确说“预训练代码链路已完成”，但我不会把它表述成“训练出了高质量模型”，因为仓库里还缺独立验证集指标、推理回归测试和 SFT 结果。下一阶段先验收预训练 checkpoint 与生成，再实现 assistant-only 的 SFT 数据和 full SFT。

**追问 1：为什么不直接调用 Hugging Face Trainer？** 我选择原生 PyTorch 是为了暴露 Trainer 会替我隐藏的状态边界，例如 loss 何时除以累积步数、何时 unscale 再裁剪、optimizer 和 scaler 如何恢复、DDP 下谁负责保存等。这个选择的代价是工程代码更多，也更容易遗漏边界，但学习收益是我能解释每一步更新对应哪些 batch、checkpoint 保存了什么，以及中断后哪些状态能真正延续。等这些机制被验证后，再迁移到成熟框架，我也能判断默认配置是否适合，而不只是会填参数。

**追问 2：如何证明不是照抄？** 我不会用“代码长得不一样”证明理解，而会用可验证任务：给一个 batch 手算 label shift；解释为什么 padding 是 `-100`；让梯度累积遇到不足整除的尾批次并说明如何 flush；中断训练后加载 optimizer、scaler 和 step；比较 cache on/off 的 greedy 输出；再逐 token 展示 SFT 中仅 assistant 片段参与 loss。这些实验能把抽象概念映射到实际行为。当前仓库已有多项实现，但部分验证仍待补，因此我会清楚区分已实现、已验证和计划验证。

#### 2. 预训练在优化什么目标？

**第一人称口播：** 这个阶段做的是 causal language modeling，也就是给定当前位置之前的 token，最大化下一个 token 的条件概率。实现上模型输出每个位置对整个词表的 logits，但第 `t` 个 logits 应与第 `t+1` 个 token 对齐，所以我去掉 logits 的最后一位、去掉 labels 的第一位后计算交叉熵。padding 的 label 被设为 `-100`，借助交叉熵的 ignore index 排除。这样每条普通文本都能直接产生监督信号，不依赖人工问答标注。训练 loss 下降表示拟合目标在改善，但不等于语言质量、知识或指令遵循已经达到可用水平。

**追问 1：为什么输入和 labels 看起来相同？** 数据集返回的 labels 确实是 input ids 的克隆，这并不是让模型复制当前 token，因为真正的错位发生在模型 loss 内部：位置零的输出预测位置一，位置一预测位置二。克隆的好处是数据接口简单，并且可以原地把 padding 改成 `-100` 而不污染输入。我要特别检查短序列、只有 BOS/EOS 的序列以及截断边界，确保右移后仍有有效监督 token；否则可能得到全忽略标签甚至无意义的 loss，这属于数据契约而不是优化器问题。

**追问 2：训练 loss 能代表模型好吗？** 不能。训练 loss 同时受 tokenizer、数据长度、重复样本和模型容量影响，而且下降可能只是记忆训练集。我会至少切出不参与更新的验证文本，报告 held-out NLL 或 PPL，同时保留固定续写 prompts 检查输出、EOS 和重复模式。跨 tokenizer 比 PPL 也不完全公平，因为 token 粒度不同，所以更严谨时可补充按 byte 归一化的指标。当前项目尚未记录这些数值，因此我会写“待测”，不会从一条漂亮生成样例推导整体质量。

#### 3. 预训练和 SFT 的本质区别是什么？

**第一人称口播：** 两个阶段底层都在做 teacher forcing 的 next-token cross entropy，模型结构和大部分训练循环可以完全复用。真正不同的是监督分布和标签边界：预训练对普通文本中除 padding 外的 token 都学习，SFT 先把 system、user、assistant 多轮消息按 chat template 串起来，通常只让 assistant 回答片段参与 loss。SFT 还从预训练权重开始、使用更低学习率，目标是把已经具备续写能力的 base model 调整为遵循指令的 chat model。因此我下一步最重视的是 SFTDataset 和 mask，而不是机械复制第二份训练循环。

**追问 1：为什么 user token 不算 loss？** user 和 system token 仍然作为上下文进入注意力，影响 assistant 的预测，只是不作为要模仿的目标。如果对它们也计算 loss，模型会同时被要求生成用户问题与系统模板，有限容量会被分散，还可能在推理时更倾向复述对话。assistant-only 并非唯一合法方案，但它与“给定上下文生成回答”的任务边界最一致。我会逐 token 解码并标注 label，确认多轮对话的每个 assistant 区间都被覆盖，同时 user、system 和 padding 都是 `-100`。

**追问 2：SFT 为什么用更低学习率？** 预训练从随机初始化开始，需要较大步长快速塑造表示；SFT 则是在已有语言能力上改变行为分布，步长过大更容易破坏已有能力或对较小指令集过拟合。官方当前脚本也采用比预训练低得多的默认学习率，但我不会把某个数值视为定律。我会用短跑实验观察训练与验证 loss、固定续写是否明显退化，并根据 batch、数据规模和有效 token 数调整，而不是只因为官方配置如此就原样复制。

### 主题二：数据与监督边界

#### 4. 你的 PretrainDataset 如何工作？

**第一人称口播：** 数据源是每行含 `text` 的 JSONL。我用 datasets 库按样本读取，将文本在不自动添加特殊 token 的前提下编码，并预留两个位置手动加入 BOS 和 EOS；超过最大长度时截断，不足则补 pad。input ids 转成 long tensor，labels 从输入克隆后把 pad 位置改为 `-100`。这个实现优点是直观，适合学习和逐条调试；代价是固定长度 padding 可能浪费计算，而且逐样本 tokenize 也未必达到最高吞吐。后续我会先保证正确性，再考虑按长度分桶、packing 或离线 tokenization。

**追问 1：为什么手动加 BOS/EOS？** 因为我在 tokenizer 调用时关闭了自动特殊 token，手动添加能让序列结构和长度预算完全显式：正文最多占 `max_length-2`，两端各占一个特殊 token。风险是 tokenizer 配置如果没有合法的 bos、eos 或 pad id，或者 chat template 使用不同边界，就会产生训练推理不一致。因此初始化时应断言这些 id 存在，并用 decode 检查一条样本。对于 SFT，我会以 chat template 产出的协议为准，不能简单沿用普通文本的拼接方式。

**追问 2：固定 padding 有什么问题？** 当样本长度差异很大时，每条都补到最大长度会让注意力和前馈在 pad 上浪费算力；虽然 loss 忽略 pad，但前向计算仍然发生。可选优化包括动态 padding、按长度分桶和把多个短文档 packing 到一个序列。packing 又会引入文档边界与跨文档注意力问题，需要 EOS 或 block-diagonal mask 约束。这个学习项目现阶段采用固定长度换取接口简单，我会先统计有效 token 比例，再决定优化是否值得，不能只凭直觉声称吞吐提升。

#### 5. 你准备怎样实现 SFTDataset？

**第一人称口播：** 我会把工作拆成序列化和监督掩码两部分。序列化阶段用 tokenizer 的 chat template 处理 conversations，保留 system、user、assistant 的角色边界；监督阶段定位每个 assistant 起始标记和结束标记，仅把回答及明确选择包含的 EOS 对应 label 设成真实 token，其余位置保持 `-100`。随后统一截断和 padding。实现完成后不会马上跑大训练，而是用一条两轮样本逐 token 打印输入、下一 token 与 label，断言至少有有效监督、user 不被监督、截断时不会越界，再做几十条样本的过拟合测试。

**追问 1：用字符串查 assistant 标记可靠吗？** 直接在渲染后的字符串里找子串可能受空格、换行和 tokenizer 切分影响，所以我更倾向先把 assistant 边界文本单独 tokenize 成 token id 模式，再在完整 token 序列中匹配。即便如此，模板升级或回答内容意外包含相同标记仍是风险。更稳的方案是 tokenizer 支持 generation mask 时直接使用模板返回的 assistant mask，或逐消息增量编码并记录区间。当前 tokenizer 能力需要实测，因此第一版可以参考官方 token 模式匹配，但必须用多轮、空回答和截断用例保护。

**追问 2：截断可能造成什么 bug？** 如果截断发生在 assistant 起始标记之后、结束标记之前，仍可监督保留下来的回答前缀，但 EOS 不存在；如果起始标记本身被截断，则这段不应产生 label。更危险的是先生成 label、后截断，或者边界搜索越过最大长度，造成输入和标签错位。我会固定先得到最终 input ids 长度，再在该范围内构造 labels，并断言两者等长。还要统计全 `-100` 样本比例，因为长 system/user 上下文可能把 assistant 完全挤出窗口，这类样本应过滤或重新截断。

#### 6. 如何验证 SFT loss mask 没写错？

**第一人称口播：** 我会采用三层验证。第一层是可视化：对一条含 system、两轮 user/assistant 的样本，按 next-token 对齐打印 token、目标 token 和 label 状态，人工检查边界。第二层是断言：输入与标签等长、有效标签数大于零、padding 全忽略、指定 user 文本对应位置全忽略。第三层是行为测试：构造极小问答集反复训练，模型应该能快速过拟合 assistant 答案；如果 loss 不降，先查全忽略或错位，如果模型复述用户，再查 user 是否被监督。这比一上来跑完整数据更快定位问题。

**追问 1：EOS 是否应参与监督？** 我倾向让 assistant 的结束标记参与监督，因为模型需要学会在回答完成时终止，否则推理容易无休止续写。但这取决于 chat template 中结束标记的具体 token 序列以及生成时传入的 eos id，二者必须一致。如果模板末尾含换行和 EOS，我要明确监督范围是只含 EOS 还是连换行一起含，并通过一个短样本检查。这里不存在脱离模板的通用答案，关键是训练边界、tokenizer 配置和推理停止条件形成闭环。

**追问 2：全 `-100` 会怎样？** 如果一个 batch 的所有目标都是 `-100`，交叉熵可能产生无有效元素的问题，表现为 NaN 或没有可用梯度；如果只有部分样本全忽略，它们也白白消耗计算。我会在 Dataset 或 collate 后统计有效 token 数，对单样本全忽略选择过滤、截断策略调整或显式报错，并把有效监督 token 比例写入日志。这样 SFT 的 loss 才能与数据组成联系起来，而不是看到异常数值后误判成学习率或模型结构问题。

### 主题三：训练工程

#### 7. 解释一下你的梯度累积实现。

**第一人称口播：** 每个 micro-batch 的 loss 先除以 accumulation steps 再反向，梯度会在参数的 grad 中累加；只有 step 达到累积边界时，我才 unscale、裁剪、执行 optimizer step、更新 scaler 并清空梯度。这样近似于更大的有效 batch，又不要求一次把所有样本放进显存。epoch 末如果剩余 micro-batch 不足一个完整窗口，我额外 flush 一次，避免丢掉尾部梯度。需要注意尾窗口仍按完整累积步数缩放，梯度会偏小；更严格的实现可按实际尾窗口大小重标定，这是我后续可完善的边界。

**追问 1：有效 batch size 怎么算？** 单进程时约等于 `micro_batch × accumulation_steps`；DDP 下若每个 rank 都有独立 micro-batch，并在更新时同步梯度，则全局有效 batch 还要乘 world size。序列任务更准确的尺度是有效 token 数，因为 padding 比例会变化。学习率是否按 batch 线性缩放不是必然规则，尤其是小模型和数据分布变化时。我会记录每次 optimizer update 对应的样本数与非 padding token 数，再比较不同配置，而不是只用命令行 batch size 描述训练规模。

**追问 2：DDP 累积时每步都同步梯度吗？** 当前普通 DDP 包装下，每次 backward 都可能触发跨卡同步，即使尚未 optimizer step，正确性没问题但通信效率不理想。更高效的做法是在非更新 micro-step 使用 `no_sync()`，只在累积窗口最后一次 backward 同步。引入它时必须正确处理 epoch 尾窗口和恢复后的 step 边界，否则最后的梯度可能永远没有同步。我的当前目标先保证语义正确；如果多卡 profiling 表明通信占比明显，再实现 `no_sync` 并用单卡/多卡参数更新对照验证。

#### 8. AMP、GradScaler 和梯度裁剪的顺序为什么重要？

**第一人称口播：** autocast 让部分算子使用低精度以节省显存和提高吞吐。fp16 动态范围较小，所以 GradScaler 先放大 loss，减少小梯度下溢；bf16 指数范围更大，通常不需要 scaler。裁剪必须发生在 `unscale_` 之后，否则看到的是被放大的梯度范数，阈值没有真实含义。正确顺序是 scale 后 backward，在更新边界 unscale、clip、step、update，最后清梯度。这个顺序已经体现在训练脚本中，但收益和稳定性仍应通过同配置短跑、NaN 监控和显存峰值实测。

**追问 1：为什么 bf16 不需要 scaler？** bf16 与 fp32 有相同数量的指数位，动态范围接近 fp32，主要损失在尾数精度，因此通常不易出现 fp16 那类小梯度整体下溢。它仍然可能出现溢出、NaN 或精度问题，所以“不需要 scaler”不等于“绝对稳定”。我会检查硬件是否原生支持 bf16，并监控 loss、梯度范数和 scaler 状态。若设备不支持，回退到 fp16 加 scaler 或 fp32，而不是仅根据命令行字符串假设某种 dtype 一定可用。

**追问 2：梯度裁剪解决什么？** 全局范数裁剪限制一次更新中异常大的梯度，减少训练发散风险，尤其在长序列、异常 batch 或混合精度下有帮助。它不是修复坏数据或错误 loss 的万能开关；如果几乎每步都触发裁剪，说明阈值、学习率、初始化或数据可能需要重新检查。我会记录裁剪前梯度范数和触发比例，再判断阈值是否合理。当前脚本使用固定阈值，但没有这些观测，因此只能说机制已实现，不能声称稳定性已被量化证明。

#### 9. 你的学习率调度有什么特点？

**第一人称口播：** 当前函数把基础学习率乘以一个从约 1 平滑下降到 0.1 的余弦系数，因此更准确地说是带最小学习率下限的 cosine decay，并没有单独的线性 warmup。每个 batch step 都会重写优化器参数组的学习率。这个实现简单，但注释里若写成“warmup + 余弦”就与实际代码不完全一致，我会纠正概念。对于从随机初始化的预训练，是否需要 warmup 应通过早期 loss、梯度范数和稳定性判断；进入 SFT 后总步数和起始学习率也需要重新计算。

**追问 1：为什么 warmup 常见？** Transformer 在训练早期参数和优化器统计量尚未稳定，立即使用峰值学习率可能导致激活或梯度剧烈波动。warmup 用若干 update 把学习率从较小值抬到峰值，再进入衰减。是否必要取决于模型规模、初始化、batch 和优化器；小模型也可能受益，但不能只因为业界常用就断言必须。我会把 scheduler 的横轴改为 optimizer update 而非 micro-step，并做有无 warmup 的短跑对比，关注早期 NaN、loss 抖动和最终验证 loss。

**追问 2：梯度累积后 scheduler 应按什么 step？** 如果学习率的语义是每次参数更新改变一次，那么应该按 optimizer update 计数，而不是每个 micro-batch 计数。当前脚本按 loader step 计算，这意味着增大 accumulation steps 会在相同更新次数下更快走完调度。它不一定导致程序错误，却会让实验不可比。我会把 global update step 单独建模，并在 checkpoint 中保存；短期学习复刻可以先保持官方风格，但必须能指出这个取舍，而不是把 micro-step 和 update-step 混为一谈。

### 主题四：分布式与恢复

#### 10. DDP 下如何避免各卡读到相同数据？

**第一人称口播：** 初始化进程组后，每个进程绑定自己的 local rank 设备，Dataset 可以共享逻辑，但由 DistributedSampler 按 rank 划分索引。每个 epoch 调用 sampler 的 `set_epoch`，使所有 rank 基于同一轮次产生一致的全局洗牌，再各自取得互不重叠的分片。反向传播时 DDP 同步梯度，因此每个进程的模型参数保持一致。日志和落盘限制在主进程，避免重复输出和文件竞争。我仍需要补一个小规模索引检查，确认 world size 下分片覆盖和重复符合 sampler 对齐策略。

**追问 1：为什么每个 epoch 要 set_epoch？** DistributedSampler 的洗牌依赖内部 epoch 值。如果始终不设置，多个 epoch 可能重复相同顺序，降低数据随机性。设置时各 rank 必须使用相同 epoch，才能保证它们从同一全局排列中切片，否则可能出现重叠或遗漏。我当前先调用 sampler 的 `set_epoch`，又设置了进程随机种子；两者负责的随机源并不完全相同。验证时我会直接打印小数据集各 rank 的索引集合，而不是仅凭 loss 正常就认为采样一定正确。

**追问 2：只让主进程保存够吗？** 对标准 DDP，每个 rank 的模型参数在同步更新后应一致，因此模型权重由主进程保存通常足够，优化器状态也应等价。但保存过程会耗时，其他 rank 若继续进入下一步，可能在同步点等待；必要时可以在保存前后加 barrier，或采用异步/分片 checkpoint。更重要的是所有 rank 对保存条件必须保持一致，不能让主进程跳过一次 collective。当前保存发生在前向反向完成之后，基本语义明确，但我会用多卡 smoke test 验证无死锁。

#### 11. 你的 checkpoint 为什么分两种？

**第一人称口播：** 推理或进入下一训练阶段只需要模型 state dict，所以我保存一份较轻的权重到输出目录，并转换到 CPU 半精度减小体积。真正的断点续训还需要 optimizer 的动量统计、GradScaler、epoch、step、world size 和实验追踪标识，因此另存完整 resume checkpoint。写完整 checkpoint 时先落到临时文件，再用原子替换覆盖目标，降低进程中断留下半个文件的风险。这种分层让“加载预训练权重做 SFT”和“恢复预训练现场”成为两个清晰接口。

**追问 1：为什么不能只保存模型权重续训？** 只恢复模型参数虽然可以继续反向，但 AdamW 的一阶、二阶矩估计会丢失，学习率调度位置也可能重置，fp16 scaler 的动态状态消失，数据可能从头重复。训练轨迹因此发生明显跳变，不能称为无损恢复。学习型项目可以允许 warm restart，但必须明确语义。如果目标是可靠 resume，我会同时保存 update step、随机数状态和 sampler 相关信息；当前实现已覆盖主要状态，随机数精确复现仍是可改进项。

**追问 2：跨 world size 恢复只换算 step 就够吗？** 不一定。按 world size 比例换算 step 可以近似保持已消费样本数，但 batch size、DistributedSampler padding、梯度累积和随机顺序都可能变化；优化器状态本身可以加载，却不保证数值轨迹连续。更严谨的做法是记录已消费样本或 update 数、全局 batch 配置和 sampler 状态，并把跨 world size 恢复定义为近似恢复而非 bitwise reproducible。当前代码有 step 换算机制，我会在文档中明确其边界，避免过度承诺。

#### 12. 断点续训有哪些容易遗漏的边界？

**第一人称口播：** 除了模型和优化器，常被遗漏的有 scaler、学习率调度位置、随机数状态、当前 epoch 内 batch 位置以及梯度累积窗口。我的实现用跳批 sampler 从已记录 step 后继续，但如果 checkpoint 恰好落在累积窗口中间，未更新的梯度并没有保存；恢复后直接按原 step 取模还可能用不足数量的 micro-batch 完成下一次更新。因此保存最好落在 optimizer update 边界，或额外保存梯度与窗口计数。epoch 末 step 的语义也要统一，避免恢复后空跑同一 epoch 或重复样本。

**追问 1：如何测试 resume 正确？** 我会做一个可控实验：固定种子和极小数据，让基线连续训练若干 update；另一条路径在某个更新边界保存、退出、恢复，再训练到相同 update。随后比较模型参数、optimizer 状态、学习率和已消费样本。严格确定性环境下应接近甚至完全一致；若使用某些非确定性 CUDA kernel，则至少 loss 轨迹和参数差异应在可解释范围。还要单独测试在 epoch 尾部和改变 world size 时的语义，而不是只验证文件能加载。

**追问 2：保存 fp16 权重会影响继续训练吗？** 如果 resume checkpoint 中模型也被压成 fp16，再加载到训练模型，原本 fp32 参数精度已经丢失，长期训练可能与保存 fp32 master weights 的轨迹不同。轻量推理权重转半精度是合理取舍，但完整 resume 更稳妥的做法是保留训练精度的模型状态，哪怕文件更大。当前工具函数对 state dict 做了 half 转换，这一点适合学习时重点审视。我会将“发布权重”和“高保真恢复状态”的精度策略分开，并用连续/恢复对照实验评估差异。

### 主题五：模型结构与推理

#### 13. 这个模型的注意力有什么特点？

**第一人称口播：** 模型采用 decoder-only causal attention，并使用 GQA：查询头数量多于键值头数量，一个 KV 头服务一组 Q 头，从而减少 KV Cache 的存储和读取。Q、K 投影后还做 RMSNorm，再施加 RoPE，把位置信息编码进旋转后的查询和键。训练或 prefill 在条件满足时使用 PyTorch 的 scaled dot product attention 快路径；有缓存或自定义 mask 时走显式注意力分支。这个组合兼顾了现代 LLM 的结构和可读性，但两条路径需要数值一致性测试，尤其是 causal mask 与 cache 偏移。

**追问 1：GQA 相比 MHA 和 MQA 如何取舍？** MHA 每个 Q 头有独立 K/V，表达能力强但 cache 最大；MQA 所有 Q 头共享一组 K/V，cache 最省但共享程度最高；GQA 位于两者之间，让若干 Q 头共享一组 K/V。当前配置中查询头和键值头的比例决定复制倍数，前向计算时将 K/V 逻辑扩展到查询头数。它的实际速度收益依赖推理 kernel 和内存带宽，不应只凭张量数量下结论，我会测不同上下文长度的 cache 占用与 tokens/s。

**追问 2：RoPE 与 KV Cache 如何配合？** 历史 K 在写入 cache 前已经按当时绝对位置旋转，新 token 必须使用从 past length 开始的位置切片，不能又从零开始。当前实现从第一层 cache 的序列长度得到 start position，再切对应 cos/sin。验证时我会把同一完整序列一次前向得到的最后位置 logits，与逐 token 使用 cache 的 logits 比较。如果偏差明显，优先检查位置偏移、causal mask 和 attention mask 长度，而不是先怀疑采样过程。

#### 14. KV Cache 为什么能加速，代价是什么？

**第一人称口播：** 自回归生成每次只新增一个 token。如果不缓存，每一步都会对整个历史重新计算每层的 K 和 V；有 cache 后，历史 K/V 直接复用，只对新 token 做投影并与历史注意力交互，因此减少重复计算。代价是每层都要保存随 batch、序列长度、KV 头数和 head dim 线性增长的缓存，占用显存并增加内存访问。我的生成实现根据 cache 长度只喂未处理 token，并更新位置偏移。下一步会在 greedy 下比较 cache 开关的 token 输出，再测不同长度的峰值显存和吞吐。

**追问 1：为什么先比较 greedy 输出？** temperature、top-k、top-p 和 multinomial 会引入随机性，即使 logits 只有微小浮点差异也可能采到不同 token，无法直接判断 cache 是否正确。greedy 每次取 argmax，配合 eval 模式、固定输入与关闭 dropout，更适合作为确定性回归。若仍分叉，我会逐步比较第一步 logits、每层 cache 形状和后续位置 logits。只有正确性成立后，再用固定 generator seed 测采样行为；性能测量也应与正确性测试分开。

**追问 2：attention mask 在 cache 场景为何容易错？** 查询长度可能只有一，而键长度包含全部历史，mask 的形状和 causal 逻辑不再是简单的方阵。padding batch 中每条 prompt 长度还可能不同，单靠统一 past length 无法表达每条序列的有效位置。当前实现适合相对简单的无 padding或统一长度生成场景；若要批量服务，需要明确 attention mask 如何随新 token 扩展，并验证左 padding、右 padding与不同完成时间。否则单条测试正常，并不代表批量推理正确。

#### 15. 为什么现在不建议你直接学 DPO/PPO/GRPO？

**第一人称口播：** 偏好优化和强化学习会在 SFT 基线之上再增加参考模型、奖励或规则信号、采样 rollout、优势或偏好损失等变量。如果预训练权重加载、chat template、assistant mask、EOS 和生成缓存尚未被验证，后续异常很难归因，可能把数据 bug 当成算法问题。我的学习路线是先完成可测的 base 模型，再实现 full SFT 并建立固定指令集，之后选择 DPO 作为相对简洁的偏好学习入口；PPO 或 agentic RL 留到生成与奖励链路都稳定以后。这样每阶段只新增一个主要概念。

**追问 1：为什么 DPO 比 PPO 更适合下一步？** DPO 可以直接使用 chosen/rejected 偏好对，通过策略模型与参考模型的对数概率差构造目标，不必先训练独立 reward model，也不需要在线 rollout 和价值网络，工程变量更少。它仍要求我正确计算 assistant token 的序列 log probability、冻结参考模型并处理长度 mask，所以正好建立在 SFT 数据和 loss mask 之上。我不会在 SFT 尚未验收时启动它；先让每条偏好样本的 chosen/rejected 概率计算可手工检查，再进行训练。

**追问 2：SFT 做到什么程度才进入下一阶段？** 我会设定工程门槛而不是主观觉得“回答还行”：checkpoint 加载前后 greedy 一致；cache on/off 在短序列一致；SFT 样本的 assistant-only mask 通过多轮和截断测试；小数据能过拟合；完整训练有验证 loss 和固定 prompt 回归；输出能正确停止且没有普遍复述 user。质量阈值可根据算力设定，但上述正确性门槛不能省。当前项目尚未全部满足，所以合理下一步是补 eval 和 SFT，而不是扩展更多算法名词。

## 4. 源码证据索引

| 主题 | 关键路径与内部符号 | 对应正文位置 |
| --- | --- | --- |
| 预训练入口 | `trainer/train_pretrain.py::train_epoch`、main 参数与初始化 | 问题 1、2、7、8、9 |
| 预训练数据 | `dataset/lm_dataset.py::PretrainDataset` | 问题 4 |
| 自回归 loss | `model/model.py::PeanutMindForCausalLM.forward` | 问题 2、3 |
| 注意力与位置编码 | `model/model.py::Attention`、`apply_rotary_pos_emb`、`repeat_kv` | 问题 13 |
| 自回归生成 | `model/model.py::PeanutMindForCausalLM.generate` | 问题 14 |
| MoE 路由 | `model/model.py::MOEFeedForward` | 简历 bullet“稀疏计算探索” |
| 训练状态 | `trainer/trainer_utils.py::lm_checkpoint` | 问题 11、12 |
| 分布式与随机种子 | `trainer/trainer_utils.py::init_distributed_mode`、`setup_seed` | 问题 10 |
| 待实现 SFT | 目标位置 `dataset/lm_dataset.py::SFTDataset`、`trainer/train_full_sft.py` | 问题 3、5、6、15 |

## 5. 交给 /asu 的项目事实摘要

- **项目名称和目标岗位：** PeanutMind；目标岗位未提供，可面向大模型算法工程、LLM 训练工程或 AI 工程方向进一步裁剪。
- **个人职责边界：** 个人学习型复刻；从当前仓库提交可确认模型实现与预训练链路，不能表述为生产上线或团队 Owner。
- **关键技术动作：** decoder-only Transformer、GQA/RoPE/RMSNorm/SwiGLU、可选 MoE、预训练 Dataset、AMP、梯度累积、DDP、checkpoint、KV Cache 与采样。
- **可核验证据：** `model/model.py`、`dataset/lm_dataset.py`、`trainer/train_pretrain.py`、`trainer/trainer_utils.py` 及仓库提交记录。
- **候选表述：** “基于原生 PyTorch 从零复刻轻量 LLM 的模型、预训练和生成链路，并实现混合精度、多卡训练及断点恢复机制。”
- **待补事实：** 实际训练硬件、数据规模、训练时长、验证 loss/PPL、吞吐、峰值显存、生成样例、cache 加速比、SFT 完成状态。

## 6. 交给 /interview 的高风险 Claim 清单

- **Ownership Claim：** 仅能确认个人学习复刻；“从零主导”“独立设计架构”需根据实际参考官方代码的程度谨慎表述。
- **Metric Claim：** 当前没有可靠的 loss、PPL、速度、显存、参数规模实验记录，任何数字均需补测。
- **Architecture Claim：** 可以说实现了模型与训练机制；“优化了”“提升了”需有对照实验。MoE、DDP、resume 和 KV Cache 尤其需要边界测试。
- **Result Claim：** 当前不能声称模型达到可用对话质量、上线、被采用或优于官方；SFT 尚未在仓库中实现。
