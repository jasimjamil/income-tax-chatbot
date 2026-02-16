import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

st.set_page_config(page_title="Income Tax Chatbot", page_icon="🏛️")
st.title("🏛️ Income Tax Chatbot")
st.caption("TinyLlama fine-tuned on Income Tax Act · Section 8.1")

# ── Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens  = st.slider("Max new tokens",  50, 400, 150)
    temperature = st.slider("Temperature",     0.1, 1.2, 0.7, 0.05)
    top_p       = st.slider("Top-p",           0.5, 1.0, 0.9, 0.05)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Load model (cached so it only loads once)
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    repo_id   = "muhammadjasim12/incometaxcassendra"
    token     = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=token)
    base      = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_4bit=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, repo_id, use_auth_token=token)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
st.success("✅ Model loaded and ready!")

# ── Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Input & generation
if prompt := st.chat_input("Ask about Income Tax Act Section 8.1…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            formatted = (
                f"<|system|>\nYou are an income tax expert.\n"
                f"<|user|>\n{prompt}\n<|assistant|>\n"
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                )
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


        