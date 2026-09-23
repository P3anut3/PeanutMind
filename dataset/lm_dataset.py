from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

# 关闭 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多 worker 下出现死锁或刷屏警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。", "你是peanutmind，一个小巧但有用的语言模型。", "你是一个专业的AI助手，请提供有价值的回答。", "你是peanutmind，请尽力帮助用户解决问题。", "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.", "You are peanutmind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.", "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

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
            add_special_tokens=False,  # 不自动加 bos/eos 等特殊 token
            max_length=self.max_length - 2,  # 截断到 max_length-2（预留 2 个位置）
            truncation=True,
        ).input_ids
        tokens = ([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id])  # 添加特殊[bos]和[eos]
        # 手动补齐到 max_length：不够的部分用 pad_token_id 填充
        # [pad] * (max_length - len(tokens)) 就是把 pad 重复若干次，再拼到 tokens 后面
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # 转成 long 类型的张量（token id 必须是整数，torch 里约定用 int64/long）
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 标签 = 输入本身（语言模型的训练目标就是"预测下一个 token"，标签就是原序列）
        # .clone() 复制一份，这样下面改 labels 不会反过来影响 input_ids
        labels = input_ids.clone()
        # 把 padding 位置的标签设成 -100：交叉熵里 ignore_index=-100 会跳过这些位置，
        # 也就是"padding 不参与损失计算"
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels


class SFTDataset(Dataset):

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string')
            }]
        })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
