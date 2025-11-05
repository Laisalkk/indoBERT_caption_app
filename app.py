import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "laisalkk/indoBERT-caption"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

def generate_caption(judul, isi, label):
    prompt = f"Judul: {judul}\nIsi: {isi}\nLabel: {label}\nCaption:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.8
        )
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

st.set_page_config(page_title="IndoBERT Caption Generator", page_icon="📰", layout="centered")

st.title("📰 IndoBERT Caption Generator")
st.markdown("Model pembuat caption berdasarkan judul, isi berita, dan label (Fakta/Hoaks).")

judul = st.text_input("🧾 Judul Berita:")
isi = st.text_area("📄 Isi Berita:")
label = st.selectbox("🏷️ Label:", ["Fakta", "Hoaks"])

if st.button("🚀 Hasilkan Caption"):
    if judul.strip() and isi.strip():
        with st.spinner("Model sedang membuat caption..."):
            caption = generate_caption(judul, isi, label)
        st.success("**Caption yang dihasilkan:**")
        st.write(caption)
    else:
        st.warning("Masukkan judul dan isi berita terlebih dahulu!")

st.markdown("---")
st.caption("Ditenagai oleh IndoBERT Caption - Model oleh @laisalkk 🤗")
