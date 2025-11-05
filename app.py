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
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

def generate_caption(label, title, isi):
    prompt = f"Label: {label}. Judul: {title}. Isi: {isi}."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=380,
            num_beams=5,
            length_penalty=1.5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption.strip()

st.set_page_config(page_title="IndoBERT Caption Generator", page_icon="🧠")

st.title("🧠 IndoBERT Caption Generator")
judul = st.text_input("Masukkan Judul Berita:")
isi = st.text_area("Masukkan Isi Berita:")
label = st.selectbox("Label Berita:", ["Fakta", "Hoaks"])

if st.button("🚀 Generate Caption"):
    if judul.strip() and isi.strip():
        with st.spinner("Sedang menghasilkan caption..."):
            caption = generate_caption(label, judul, isi)
        st.success("Caption yang dihasilkan:")
        st.write(caption)
    else:
        st.warning("Lengkapi semua input dulu ya!")
