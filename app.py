import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "laisalkk/indoBERT-caption"  # model caption kamu di Hugging Face

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.set_page_config(page_title="IndoBERT Caption Generator", page_icon="📝")
st.title("📝 IndoBERT Caption Generator")
st.write("Masukkan teks atau deskripsi untuk menghasilkan **caption otomatis berbahasa Indonesia.**")

text_input = st.text_area("Masukkan kalimat deskripsi:", "", height=150)

if st.button("Generate Caption"):
    if not text_input.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menghasilkan caption..."):
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(
                **inputs,
                max_length=30,
                num_beams=5,
                repetition_penalty=2.5,
                early_stopping=True
            )
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("**Caption:** " + caption)
