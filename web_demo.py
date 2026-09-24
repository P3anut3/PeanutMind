import random
import torch
import streamlit as st
from threading import Thread

# 提前在单线程阶段预热 transformers 的 lazy 加载命名空间：
# transformers 的 _LazyModule 在多线程下偶发 "cannot import name 'AutoTokenizer'" 竞态，
# 这里强制解析一次并缓存到模块 __dict__，后续 `from transformers import ...` 就变成直接取属性。
import transformers  # noqa: F401
_ = transformers.AutoTokenizer
_ = transformers.TextIteratorStreamer
from transformers import AutoTokenizer, TextIteratorStreamer

from model.model import PeanutMindConfig, PeanutMindForCausalLM
from trainer.trainer_utils import setup_seed

st.set_page_config(page_title="PeanutMind 对话", page_icon="🥜", layout="centered")


@st.cache_resource(show_spinner=False)
def load_model(weight, hidden_size, num_hidden_layers, use_moe, device):
    """加载模型 + tokenizer（st.cache_resource 保证只加载一次，不会每次交互都重载）"""
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = PeanutMindForCausalLM(
        PeanutMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            use_moe=use_moe,
        )
    )
    moe_suffix = "_moe" if use_moe else ""
    ckp = f"./out/{weight}_{hidden_size}{moe_suffix}.pth"
    model.load_state_dict(torch.load(ckp, map_location="cpu"), strict=True)
    model.half().eval().to(device)
    return model, tokenizer


def main():
    st.title("🥜 PeanutMind 对话")
    st.caption("MiniMind 复现 · 63.9M 参数小模型")

    with st.sidebar:
        st.header("⚙️ 配置")
        weight = st.selectbox("权重", ["full_sft", "pretrain"], index=0)
        hidden_size = st.number_input("hidden_size", value=768, step=64)
        num_hidden_layers = st.number_input("num_hidden_layers", value=8, step=1)
        use_moe = st.checkbox("use_moe", value=False)

        cuda_count = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(cuda_count)] + ["cpu"]
        default_device = "cuda:3" if "cuda:3" in devices else devices[0]
        device = st.selectbox("device", devices, index=devices.index(default_device))

        temperature = st.slider("temperature", 0.0, 2.0, 0.85, 0.05)
        top_p = st.slider("top_p", 0.0, 1.0, 0.95, 0.05)
        max_new_tokens = st.slider("max_new_tokens", 64, 4096, 1024, 64)
        open_thinking = st.checkbox("开启思考（先 <think> 再回答）", value=False)
        historys = st.slider("携带历史轮数", 0, 20, 4, 2)

        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()

    model, tokenizer = load_model(
        weight, hidden_size, num_hidden_layers, use_moe, device
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("输入消息…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 只带最近 historys 轮历史
    history = st.session_state.messages[:-1]
    if historys:
        history = history[-historys:]
    conversation = history + [{"role": "user", "content": prompt}]

    if weight == "pretrain":
        text = tokenizer.bos_token + prompt
    else:
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=open_thinking,
        )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    setup_seed(random.randint(0, 31415926))

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,
        streamer=streamer,
    )

    with st.chat_message("assistant"):
        # 子线程跑 generate 往 streamer 里塞 token，主线程用 write_stream 边生成边显示
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        full = st.write_stream(streamer)
        thread.join()

    st.session_state.messages.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
