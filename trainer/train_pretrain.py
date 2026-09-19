import os
import sys

# 下面两个 hack 是为了让脚本能直接 `python trainer/train_pretrain.py` 运行：
# 脚本位于 trainer/ 子目录，默认 sys.path 里没有项目根目录，
# 手动把上一级目录加进去，才能 import model / dataset / trainer 这些包
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 把工作目录切到脚本所在的 trainer/ 目录：
# 下面代码里大量的 "../xxx"（如 ../model、../dataset、../out）都是相对路径，
# 只有 CWD 在 trainer/ 下才能正确指到项目根目录。加了这一行后，
# 不管你在哪个目录下用 `python trainer/train_pretrain.py` 启动，路径都能跑对。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import datasets  # noqa: F401  # Windows 下 pyarrow/torch 的 DLL 冲突 workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import PeanutMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 注意：args / model / optimizer / scaler / autocast_ctx / lm_config 都是下面
    # `if __name__ == "__main__"` 里定义的全局变量，本函数被调用时它们已经就绪。
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 把数据搬到训练设备（GPU/CPU）上
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        # 按当前 step 计算学习率（warmup + 余弦退火等策略，见 trainer_utils.get_lr）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 把新学习率写到优化器的每个参数组里（lr 是可以逐 step 动态改的）
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 混合精度：autocast 让部分计算用 bf16/fp16 做，省显存、提速
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 总损失 = 语言模型损失 + MoE 路由辅助损失
            loss = res.loss + res.aux_loss
            # 梯度累积：loss 除以累积步数，等价于"多步梯度求平均"
            loss = loss / args.accumulation_steps

        # scaler.scale：先把 loss 放大再反向，避免 fp16 下梯度太小下溢成 0
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步才真正更新一次参数（梯度累积）
        if step % args.accumulation_steps == 0:
            # unscale 还原梯度真实尺度，再做梯度裁剪（防止梯度爆炸）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)  # 用累积的梯度更新参数
            scaler.update()         # 动态调整 scaler 的缩放因子

            # 清空梯度；set_to_none=True 直接把 grad 置 None，比置 0 更省显存
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # loss 在累积时被除以过 accumulation_steps，这里乘回来还原"真实"损失
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 保存前要把模型"解包"：
            # 1) DDP 把模型包了一层，真正的模型在 model.module
            # 2) torch.compile 也会包一层，真正的模型在 _orig_mod 属性里
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 权重转 fp16 再存 CPU，减小 checkpoint 体积
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict
        # 及时释放本 step 的中间变量，避免显存堆积
        del input_ids, labels, res, loss

    # epoch 结束时，如果还有不足 accumulation_steps 的"剩余梯度"，也要 flush 掉更新一次
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len",
        default=340,
        type=int,
        help="训练的最大截断长度（中文1token≈1.5~1.7字符）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/pretrain_t2t_mini.jsonl",
        help="预训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="基于哪个权重训练，为none则从头开始",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名"
    )
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用torch.compile加速（0=否，1=是）",
    )
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 初始化多卡 DDP，返回本进程的 rank
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 多卡时每个进程用自己那张卡
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = PeanutMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    # 若开启续训，先从 checkpoint 里读回训练状态（模型/优化器/step 等）
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # autocast 上下文：CPU 上没有混合精度，用 nullcontext()（什么都不做）
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 这里其实用的是 swanlab（国产，接口兼容 wandb）

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 多卡时用 DistributedSampler，让每张卡只拿到一部分不重叠的数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler：fp16 训练时用来做梯度缩放（bf16 不需要，所以 enabled 只在 fp16 时开）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)  # 图模式加速
        Logger("torch.compile enabled")
    if dist.is_initialized():
        # DDP：把模型包一层，多卡间自动同步梯度
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每个 epoch 重新设采样器的随机状态（DDP 下保证各卡采样顺序一致、且不同 epoch 不同）
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch)
        # randperm：生成 0..N-1 的随机排列，作为"洗牌后的样本索引"
        indices = torch.randperm(len(train_ds)).tolist()
        # 续训时，跳过已经训练过的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or indices, args.batch_size, skip
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,  # 多进程并行加载数据
            pin_memory=True,               # 把数据锁在内存页，加速 CPU->GPU 拷贝
        )
        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()  # 等待所有进程到齐
        dist.destroy_process_group()
