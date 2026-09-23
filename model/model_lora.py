import torch
from torch import nn


# 定义 LoRA 网络结构：ΔW = B·A（一个低秩修正量）
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA 的秩，控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)    # 降维 in -> rank
        self.B = nn.Linear(rank, out_features, bias=False)   # 升维 rank -> out
        # 矩阵 A 高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵 B 全零初始化（这样一开始 ΔW = B·A = 0，不影响原模型输出）
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))   # = B·(A·x) = (B·A)·x = ΔW·x


def apply_lora(model, rank=16):
    """给模型中 in_features == out_features 的 Linear 层挂上 LoRA，
    并把该层 forward 改成「原 Linear 输出 + LoRA 输出」。"""
    # 先收集目标层，再统一修改：避免在遍历 modules() 的过程中往模块里加子模块
    targets = [
        module
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.in_features == module.out_features
    ]
    for module in targets:
        lora = LoRA(module.in_features, module.out_features, rank=rank).to(
            module.weight.device
        )
        # 把 LoRA 挂到该 Linear 上（作为子模块，A/B 的参数会自动登记进 model.parameters()）
        setattr(module, "lora", lora)
        # 保存该层「原始」的 forward（bound method，绑定到当前 module）
        original_forward = module.forward

        # 新 forward = 原 Linear 输出 + LoRA 输出
        # 用默认参数 _orig/_lora 捕获当前值，避免闭包晚绑定
        # （否则循环里所有层最后都会引用到最后一个 lora）
        def forward_with_lora(x, _orig=original_forward, _lora=lora):
            return _orig(x) + _lora(x)

        module.forward = forward_with_lora


def load_lora(model, path):
    """从 checkpoint 加载 LoRA 参数（只加载 LoRA 部分，不动原权重）。"""
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    # 兼容 DDP 保存时带 "module." 前缀的情况
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."
            # 从完整 state_dict 里抽出该层 LoRA 的 A.weight / B.weight
            lora_state = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            module.lora.load_state_dict(lora_state)


def merge_lora(model, lora_path, save_path):
    """把 LoRA 合并回原权重：W_merged = W + B·A，得到不带 LoRA 的完整权重并保存。
    注意：调用前需先 apply_lora(model) 挂上 LoRA 结构。"""
    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    # 先取完整 state_dict（排除 lora 子模块），再对挂过 LoRA 的层覆盖成合并后的权重
    state_dict = {
        k: v.cpu().half()
        for k, v in raw_model.state_dict().items()
        if ".lora." not in k
    }
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            # ΔW = B @ A，形状 [out, in]，和 W 一致，直接相加
            merged = (
                module.weight.data + module.lora.B.weight.data @ module.lora.A.weight.data
            )
            state_dict[f"{name}.weight"] = merged.cpu().half()
    torch.save(state_dict, save_path)
