import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import os

MODEL_NAME = "laisalkk/indoBERT-caption"

@st.cache_resource
def load_model():
    # Ambil token Hugging Face dari Streamlit Secret (kalau ada)
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        login(token=hf_token)

    # Load model & tokenizer langsung dari repo kamu di Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_auth_token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device
