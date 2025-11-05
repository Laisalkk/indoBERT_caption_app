import streamlit as st
import torch
from transformers import AutoTokenizer, EncoderDecoderModel

MODEL_NAME = "cahya/indoBERT2BERT-base"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Fungsi untuk membuat ringkasan
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(**inputs, max_length=100, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Tampilan Streamlit
st.set_page_config(page_title="IndoBERT Caption Generator", page_icon="📝", layout="centered")

st.title("📝 IndoBERT Caption Generator")
st.markdown("Model ringkasan teks berbasis IndoBERT2BERT oleh Cahya 🤗")

text_input = st.text_area("Masukkan teks untuk diringkas:", "")

if st.button("✨ Buat Ringkasan"):
    if text_input.strip():
        with st.spinner("Sedang menghasilkan ringkasan..."):
            summary = generate_summary(text_input)
        st.success("**Hasil Ringkasan:**")
        st.write(summary)
    else:
        st.warning("Masukkan teks terlebih dahulu!")

st.markdown("---")
st.caption("Ditenagai oleh IndoBERT2BERT (Cahya) - Hugging Face 🤗")
