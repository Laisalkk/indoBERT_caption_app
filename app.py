import os
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================================================
# ✅ Pastikan versi library sesuai dengan environment Colab
# =========================================================
os.system("pip install transformers==4.45.0 tokenizers==0.20.3 huggingface-hub==0.36.0 --quiet")

# =========================================================
# ⚙️ Konfigurasi
# =========================================================
MODEL_NAME = "laisalkk/indoBERT-caption"

# =========================================================
# 🚀 Load model dan tokenizer (cache biar cepat)
# =========================================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# =========================================================
# 🧼 Cleaning & Postprocessing
# =========================================================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?()/-]', '', text)
    return text.strip()

def normalize_capitalization(text):
    if not text:
        return ""
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    return text

def postprocess_caption(text):
    text = re.sub(r'\s*\.\s*\.', '.', text)
    text = re.sub(r'\s*-\s*', ' - ', text)
    text = re.sub(r'\s+', ' ', text)
    if not text.endswith('.'):
        text += '.'
    return text.strip()

# =========================================================
# ✍️ Fungsi generate caption
# =========================================================
def generate_caption(label, title, isi):
    isi_clean = clean_text(isi)
    title_clean = clean_text(title)
    if not isi_clean:
        return "Tidak ada isi untuk digenerate."

    prompt = f"Label: {label}. Judul: {title_clean}. Isi: {isi_clean}."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(
        **inputs,
        max_length=380,
        num_beams=5,
        length_penalty=1.5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    caption = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    caption = normalize_capitalization(postprocess_caption(caption))
    return caption

# =========================================================
# 💻 Streamlit UI
# =========================================================
st.title("🧠 IndoBERT Caption Generator (laisalkk/indoBERT-caption)")
st.write("Masukkan **label**, **judul**, dan **isi berita/artikel** untuk menghasilkan caption otomatis.")

label = st.text_input("🪶 Label")
title = st.text_input("📰 Judul")
isi = st.text_area("📄 Isi Berita/Artikel")

if st.button("🚀 Generate Caption"):
    with st.spinner("Menghasilkan caption..."):
        caption = generate_caption(label, title, isi)
    st.success("✅ Caption berhasil dibuat!")
    st.write("**Hasil Caption:**")
    st.write(caption)
