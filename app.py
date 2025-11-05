import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 🟢 HARUS paling atas sebelum perintah Streamlit lainnya
st.set_page_config(page_title="IndoBERT Caption Generator", page_icon="📰")

# Fungsi memuat model dan tokenizer
@st.cache_resource
def load_model():
    MODEL_NAME = "laisalkk/indoBERT-caption"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Tampilan halaman utama
st.title("📰 IndoBERT Caption Generator")
st.markdown(
    "Masukkan **judul berita**, **label (Hoaks atau Fakta)**, dan **isi berita** "
    "untuk menghasilkan *caption otomatis berbahasa Indonesia*."
)

# Input dari pengguna
judul = st.text_input("🗞️ Judul Berita:")
label = st.selectbox("🏷️ Label Berita:", ["Hoaks", "Fakta"])
isi = st.text_area("🧾 Isi Berita:", height=200)

# Tombol generate caption
if st.button("🔍 Generate Caption"):
    if not judul.strip() or not isi.strip():
        st.warning("Masukkan judul dan isi berita terlebih dahulu.")
    else:
        with st.spinner("Sedang menghasilkan caption..."):
            prompt = f"Judul: {judul}\nLabel: {label}\nIsi: {isi}\nCaption:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                repetition_penalty=2.5,
                early_stopping=True
            )
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.subheader("📝 Caption yang Dihasilkan:")
            st.success(caption)
