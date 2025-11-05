import streamlit as st
from transformers import pipeline

MODEL_NAME = "laisalkk/indoBERT-caption"

@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME
    )
    return generator

def generate_captions(generator, judul, isi, label, num_captions=2):
    prompt = f"Judul: {judul}\nIsi: {isi}\nLabel: {label}\nCaption:"
    results = generator(
        prompt,
        max_new_tokens=60,          # ✅ perbaikan: ganti dari max_length
        num_return_sequences=num_captions,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )
    captions = [r["generated_text"].split("Caption:")[-1].strip() for r in results]
    return captions

def main():
    st.set_page_config(page_title="IndoBERT Caption Generator", layout="centered")
    st.title("🇮🇩 IndoBERT Caption Generator")
    st.write("Masukkan berita dan label Fakta/Hoaks untuk menghasilkan caption otomatis dengan model **CahyaBERT Summary**.")

    judul = st.text_input("📰 Judul Berita")
    isi = st.text_area("📄 Isi Berita", height=200)
    label = st.radio("🏷️ Label Berita", ["Fakta", "Hoaks"])
    num_captions = st.slider("🔢 Jumlah caption yang ingin dihasilkan:", 1, 5, 2)

    if st.button("🔍 Hasilkan Caption"):
        if judul.strip() and isi.strip():
            generator = load_model()
            try:
                captions = generate_captions(generator, judul, isi, label, num_captions)
                st.success(f"**{num_captions} Caption yang dihasilkan:**")
                for i, cap in enumerate(captions, 1):
                    st.markdown(f"**🟢 Caption {i}:** {cap}")
            except Exception as e:
                st.error(f"Terjadi error saat generate caption: {str(e)}")
        else:
            st.warning("Mohon isi judul dan isi berita terlebih dahulu.")

    st.markdown("---")
    st.caption("Model: laisalkk/indoBERT-caption • Base: CahyaBERT Summary • Dibuat oleh laisalkk")

if __name__ == "__main__":
    main()
