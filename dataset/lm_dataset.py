from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

# 关闭 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多 worker 下出现死锁或刷屏警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """预训练数据集：继承 torch 的 Dataset，实现 __len__ 和 __getitem__ 后就能直接丢给 DataLoader 用。"""

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 用 HuggingFace datasets 库加载一个 JSONL 文件（每行一个 {"text": ...} 这样的对象）
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        # 返回样本总数，DataLoader 据此知道一共能切多少个 batch
        return len(self.samples)

    def __getitem__(self, index):
        # 取第 index 条样本，把文本编码成 token id 列表
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,        # 不自动加 bos/eos 等特殊 token
            max_length=self.max_length - 2,  # 截断到 max_length-2（预留 2 个位置）
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] # 添加特殊[bos]和[eos]
        # 手动补齐到 max_length：不够的部分用 pad_token_id 填充
        # [pad] * (max_length - len(tokens)) 就是把 pad 重复若干次，再拼到 tokens 后面
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        # 转成 long 类型的张量（token id 必须是整数，torch 里约定用 int64/long）
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 标签 = 输入本身（语言模型的训练目标就是"预测下一个 token"，标签就是原序列）
        # .clone() 复制一份，这样下面改 labels 不会反过来影响 input_ids
        labels = input_ids.clone()
        # 把 padding 位置的标签设成 -100：交叉熵里 ignore_index=-100 会跳过这些位置，
        # 也就是"padding 不参与损失计算"
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels
