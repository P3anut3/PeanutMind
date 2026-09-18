import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class PeanutMindConfig(PretrainedConfig):
    """模型配置：集中存放所有超参数，方便保存/加载时一并序列化。"""

    model_type = "PeanutMind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.hidden_act = kwargs.get("hidden_act", "silu")
        # 中间层维度：约取 hidden_size 的 3.14 倍，再向上取整到 64 的倍数（方便硬件对齐/向量化）
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        # YaRN：一种 RoPE 外推方法，让模型在比训练时更长的上下文上也能工作
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        ### MoE 相关配置（use_moe=False 时忽略）
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get(
            "moe_intermediate_size", self.intermediate_size
        )
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    """RMSNorm（Root Mean Square 归一化）：
    LLaMA/MiniMind 用的归一化方式，比传统 LayerNorm 更简单——只做"缩放"不做"平移"
    （没有 bias），计算量更小。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 防止除零的极小值
        # nn.Parameter 把张量包装成"可训练参数"；初始化为全 1（即一开始不缩放）
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x.pow(2).mean(dim=-1, keepdim=True)：沿最后一维求"均方值"
        # torch.rsqrt = 1/sqrt，所以整体等价于 x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float32 再计算：float16 做平方求和容易精度不足，产生误差
        # .type_as(x) 算完再转回原来的 dtype（如 float16）
        return (self.weight * self._norm(x.float())).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    """预计算 RoPE（旋转位置编码）需要的 cos/sin 表。
    RoPE 不像传统方法那样"把位置向量加到 token 上"，而是把 q、k 向量按两两一组旋转，
    用旋转角度来编码位置信息。这里提前把每个位置的旋转角 cos/sin 算好，避免每次 forward 重算。"""

    # 频率：1 / (rope_base^(2i/dim))，i 从 0 到 dim/2-1，呈几何级数从 1 递减到 1/rope_base
    freqs, atten_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim)),
        1.0,
    )
    if rope_scaling is not None:
        # YaRN 外推：f'(i) = f(i)((1-γ) + γ/s)，γ 是随维度线性变化的 ramp（插值系数）
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),  # 上下文扩展倍数
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end / orig_max > 1.0:  # 目标长度超出原始训练长度时，才需要外推
            # 反解出"该频率维度"对应的原始位置，用于确定哪些频率需要缩放
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(
                math.ceil(inv_dim(beta_slow)), dim // 2 - 1
            )
            # ramp 从 0 平滑过渡到 1，决定每个频率维度缩放多少（低频几乎不缩放，高频缩放多）
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    # 每个位置 t 的旋转角 = t * freqs
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # 外积 -> [end, dim//2]
    # 把 cos/sin 各自复制一份拼起来变成 [end, dim]，
    # 让每个 (a,b) 对的两个分量用同一个角度，正好配合下面的 rotate_half
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把旋转位置编码施加到 q、k 上。
    核心：把向量看成两两一组的 (a, b)，旋转 θ 角 => (a·cosθ - b·sinθ, a·sinθ + b·cosθ)。"""

    # 旋转 [a, b] -> [-b, a]：后一半取反挪到前面，是实现上面旋转公式的关键一步
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # unsqueeze 加一维，让 cos/sin 能按"头"维度广播到 q、k 上
    q_embed = (
        (q * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ).to(q.dtype)
    k_embed = (
        (k * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    ).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA（分组查询注意力）：让一个 KV 头服务多个 Q 头，从而节省 KV 显存。
    这里把 KV 头复制 n_rep 次，使 KV 头数和 Q 头数对齐。"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # x[:, :, :, None, :]：在第 3 维后插入一个长度为 1 的新维度（等价 unsqueeze(3)）
    # .expand：广播式扩展，不真正复制内存（只改 stride），之后再 reshape 成连续形状
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: PeanutMindConfig):
        super().__init__()
        # 先确定 Q 头数和 KV 头数（GQA：KV 头通常比 Q 头少）
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个 KV 头对应几个 Q 头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True  # 因果注意力：当前 token 只能看到它之前的 token
        # 三个线性层分别算出 q、k、v（bias=False，LLaMA 系通常不用 bias）
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        # 对 q、k 做归一化（QK-Norm，能提升训练稳定性）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 是否能用 PyTorch 自带的 flash attention 实现（scaled_dot_product_attention）
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )

    def forward(
        self,
        x,
        positon_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        bs, seq_len, _ = x.shape
        # 先投影，再 view 成 [bs, seq_len, heads, head_dim] 的多头形状
        xq = self.q_proj(x).view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bs, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = positon_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 乘上位置编码
        if past_key_value is not None:
            # KV 缓存：把历史的 k、v 拼到前面，避免重复计算过去 token 的 k、v
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        # 转成 [bs, heads, seq, head_dim]，方便做矩阵乘法
        # repeat_kv 把 KV 头复制到和 Q 头一样多（GQA）
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # flash attention 快路径：训练/prefill 且无需自定义 mask 时用
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            # 手动实现的注意力：scores = q·k^T / sqrt(head_dim)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                # 因果 mask：上三角（对角线以上）设为 -inf，softmax 后为 0，阻止看到未来 token
                # 有 KV 缓存时只对"新 token"这一小段做 mask（取最后 seq_len 列）
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                # 把 padding 位置也 mask 掉（1.0 - mask 得到需屏蔽的位置，乘一个大负数）
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax 前转 float32 提升数值稳定性，再转回原 dtype
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # 两个分支最后都要过输出投影 o_proj（把多头结果拼回 hidden_size）和残差 dropout
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """标准 SwiGLU 前馈网络：gate_proj 做门控，与 up_proj 相乘，再 down_proj 投影回去。"""

    def __init__(self, config: PeanutMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # ACT2FN 是 transformers 提供的"激活函数名 -> 函数"字典，如 "silu" -> nn.SiLU
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: down( act(gate(x)) * up(x) )
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """MoE（混合专家）前馈：用一个"路由器 gate"从多个 FFN 专家中挑 top-k 个处理每个 token。
    好处是总参数量可以很大，但每个 token 只激活少数专家，计算量可控。"""

    def __init__(self, config: PeanutMindConfig):
        super().__init__()
        self.config = config
        # 路由器：把 hidden_size 映射成 num_experts 个分数（每个专家一个分数）
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 所有专家，每个专家就是一个普通的前馈网络
        self.experts = nn.ModuleList(
            [
                FeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # 展平成 [batch*seq, dim]，逐 token 处理
        # 每个 token 对每个专家的"权重"（softmax 转成概率）
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 选出权重最大的 k 个专家及其索引
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                # 把选中的 k 个权重再归一化，使它们的和为 1
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # k=1：前向用权重 1.0，但梯度仍流过 top1（straight-through 直通估计技巧）
                # detach() 切断梯度，所以 (top1 - top1.detach()) 前向为 0，加 1.0 后前向恒为 1，
                # 但反向时梯度能传回 top1
                top1 = torch.topk(
                    F.softmax(self.gate(x_flat.detach()), dim=-1),
                    k=1,
                    dim=-1,
                    sorted=False,
                )[0]
                topk_weight = top1 - top1.detach() + 1.0
        y = torch.zeros_like(x_flat)  # 累加输出
        for i, expert in enumerate(self.experts):
            mask = topk_idx == i  # 哪些 token 被路由到第 i 个专家
            if mask.any():
                # 找出被路由到该专家的 token 位置
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                # index_add_：在 y 的 token_idx 这些行上，累加"专家输出 * 权重"
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 训练时即使某专家没分到 token，也要把它接进计算图，
                # 否则它的参数收不到梯度。0 * sum(...) 恒为 0，但能让参数进入图。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 辅助损失：鼓励路由器把 token 均匀分给各专家（负载均衡）
            # one_hot(...).mean(0) = 每个专家实际分到的 token 比例
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            # 不训练时 aux_loss 为 0（new_zeros 保证和设备、dtype 一致）
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)


class PeanutMindBlock(nn.Module):
    """一个 Transformer 层：Pre-Norm + 注意力 + Pre-Norm + FFN，含残差连接。"""

    def __init__(self, layer_id: int, config: PeanutMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        # Pre-Norm：归一化放在子层之前（LLaMA 系做法，比原始 post-norm 训练更稳定）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        # 注意力 + 残差
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        # FFN + 残差
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class PeanutMindModel(nn.Module):
    def __init__(self, config: PeanutMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入表：把 token id 映射成 hidden_size 维向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 堆叠 num_hidden_layers 个 Transformer block
        self.layers = nn.ModuleList(
            [PeanutMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算 RoPE 的 cos/sin 表
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        # register_buffer：注册成 buffer（不是可训练参数，不参与梯度，但会随模型 .to(device) 一起移动）
        # persistent=False：保存/加载 checkpoint 时不保存该 buffer（加载后会变全 0，见 forward 里的判断）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        # 如果传进来的是 transformers 的 Cache 对象，这里暂不支持，直接忽略
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        # 初始时 past_key_values 为 None，用 [None]*层数 表示"每层都还没有缓存"
        past_key_values = past_key_values or [None] * len(self.layers)
        # 已生成的 token 数（看第一层缓存的序列长度），用于计算 RoPE 的位置偏移
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 查词表：[batch, seq_len] 的 id -> [batch, seq_len, dim] 的向量
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # freqs_cos/sin 是 persistent=False 的 buffer，从 checkpoint 加载后会变成全 0。
        # 用 freqs_cos[0,0]==0 判断它是否被清空过（正常计算时 freqs_cos[0,0]=cos(0)=1），
        # 若是则重新算一遍并搬到当前设备。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos, self.freqs_sin = (
                freqs_cos.to(hidden_states.device),
                freqs_sin.to(hidden_states.device),
            )
        # 根据当前生成位置，切出对应的 cos/sin 片段作为本步的位置编码
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        # 累加各层 MoE 的辅助损失；若没启用 MoE，列表为空，sum 返回初始值（0 张量）
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss


class PeanutMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = PeanutMindConfig
    # 权重绑定：lm_head 和词嵌入共享同一份权重（省参数，语言模型常用技巧）
    _tied_weight_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: PeanutMindConfig):
        self.config = config or PeanutMindConfig()
        super().__init__(self.config)
        self.model = PeanutMindModel(self.config)
        # 输出头：把最后的 hidden state 映射回词表大小，得到每个 token 的 logits
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        # logits_to_keep>0 时只保留最后几个位置的 logits（推理省显存）；=0 表示保留全部
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 语言模型是"用前一个 token 预测下一个 token"：logits[t] 应对应 labels[t+1]
            # 所以 logits 去掉最后一位，labels 去掉第一位，做错位对齐
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 展平成二维算交叉熵；ignore_index=-100 表示这些位置不算损失（如 padding）
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    # 实现参考：https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()  # 推理模式：不记录梯度，省显存提速
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):
        # 若一次生成多个序列，就把输入沿 batch 维复制 num_return_sequences 份
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = (
            attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        )
        past_key_values = kwargs.pop("past_key_values", None)
        # finished 标记每条序列是否已生成结束符 eos
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # 已生成的 token 数（有缓存时看缓存的序列长度）
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 只把新 token 喂进模型（老 token 用 KV 缓存，不用重算）
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs
            )
            # attention_mask 尾部追加一列 1（新 token 位置）
            attention_mask = (
                torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)
                if attention_mask is not None
                else None
            )
            # 取最后一步的 logits，除以 temperature（温度越高采样越随机）
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 重复惩罚：对已出现过的 token，正分除以 penalty、负分乘以 penalty，压低其概率
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )
            if top_k > 0:
                # top-k 过滤：只保留分数最高的 k 个 token，其余设为 -inf
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float("inf")
            if top_p < 1.0:
                # top-p（核采样）：按概率从高到低累加，只保留累计概率不超过 p 的那批 token
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # 把"第一个使累计概率超过 p"的 token 也保留下来（边界 token 含在内）
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # scatter 把 mask 映射回原始 token 顺序，再把被过滤的 token 设为 -inf
                logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
            # 采样：softmax 转概率，multinomial 按概率随机抽一个；do_sample=False 则直接取最大
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample
                else torch.argmax(logits, dim=-1, keepdim=True)
            )
            if eos_token_id is not None:
                # 已结束的序列，之后一直填充 eos，不再改变
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids
