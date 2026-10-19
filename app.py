import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained("distilbert_model")
    mdl = AutoModelForSequenceClassification.from_pretrained("distilbert_model")
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()

st.title("📰 Fake News Detection using DistilBERT")
st.write("News article ka title aur text paste karo.")

text = st.text_area("News article", height=250)

if st.button("Check karo"):
    if len(text.split()) < 20:
        st.warning("Kam se kam 20-30 words ka article daalo, tabhi result bharosemand hoga.")
    else:
        inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        fake_p, real_p = float(probs[0]), float(probs[1])
        if real_p > fake_p:
            st.success(f"✅ REAL news ({real_p*100:.1f}% confidence)")
        else:
            st.error(f"🚨 FAKE news ({fake_p*100:.1f}% confidence)")
        st.progress(real_p, text=f"Real: {real_p*100:.1f}%  |  Fake: {fake_p*100:.1f}%")