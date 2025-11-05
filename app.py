import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# -----------------------------
# Load model & tokenizer
# -----------------------------
MODEL_NAME = "laisalkk/indoBERT-caption"  # ganti sesuai model Hugging Face kamu

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -----------------------------
# Cleaning & postprocess
# -----------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'[^\w\s.,;:!?()/-]', '', text)
    return text.strip()

def normalize_capitalization(text):
    if not text:
        return ""
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    replacements = {
        "bpjs": "BPJS", "bri": "BRI", "bni": "BNI", "ojk": "OJK",
        "ri": "RI", "who": "WHO", "pt ": "PT ", "pmi": "PMI",
        "puskesmas": "Puskesmas", "indomaret": "Indomaret",
        "telkomsel": "Telkomsel", "facebook": "Facebook",
        "twitter": "Twitter", "whatsapp": "WhatsApp",
        "jokowi": "Jokowi", "risma": "Risma"
    }
    for k, v in replacements.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text

def postprocess_caption(text):
    text = re.sub(r'\s*\.\s*\.', '.', text)
    text = re.sub(r'\s*-\s*', ' - ', text)
    text = re.sub(r'\s+', ' ', text)
    def cap_after_period(s):
        return re.sub(r'(?<=[.!?]\s)([a-z])', lambda m: m.group(1).upper(), s)
    text = cap_after_period(text)
    if not text.endswith('.'):
        text += '.'
    return text.strip()

# -----------------------------
# Caption generator
# -----------------------------
def generate_caption(label, title, isi, num_captions=2):
    isi_clean = clean_text(isi)
    title_clean = clean_text(title)
    prompt = f"Label: {label}. Judul: {title_clean}. Isi: {isi_clean}."

    captions = []
    for _ in range(num_captions):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
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
        captions.append(caption)
    return captions

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Auto Caption Generator – IndoBERT2BERT")
st.write("Masukkan judul berita, isi, dan label (Fakta/Hoaks), lalu model akan menghasilkan caption otomatis.")

judul = st.text_input("📰 Judul Berita")
isi = st.text_area("📄 Isi Berita")
label = st.selectbox("🏷️ Label", ["Fakta", "Hoaks"])
num_captions = st.slider("🔢 Jumlah caption yang dihasilkan", 1, 3, 2)

if st.button("🚀 Generate Caption"):
    with st.spinner("Sedang menghasilkan caption..."):
        captions = generate_caption(label, judul, isi, num_captions)
    st.success("✅ Caption berhasil dihasilkan!")
    for i, cap in enumerate(captions, 1):
        st.markdown(f"**🟢 Caption {i}:** {cap}")
