import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_NAME = "laisalkk/indoBERT-caption"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs.squeeze().tolist()

st.set_page_config(page_title="IndoBERT Caption Classifier", page_icon="🤖")
st.title("🧠 IndoBERT Caption Classifier")

input_text = st.text_area("Masukkan teks untuk diklasifikasi:")

if st.button("Prediksi"):
    if input_text.strip():
        with st.spinner("Memproses..."):
            pred, probs = predict_label(input_text)
            st.success(f"Label Prediksi: {pred}")
            st.write(f"Probabilitas Tiap Label: {probs}")
    else:
        st.warning("Masukkan teks dulu ya!")
