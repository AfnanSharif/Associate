
import streamlit as st
import joblib
import yaml

st.set_page_config(page_title="Associate | NLP", layout="wide")
st.title("📝 Associate: NLP Analyzer")

try:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = joblib.load(config['data']['model_path'])
except:
    st.error("Please run `python src/train.py` first.")
    st.stop()

text = st.text_area("Enter text to analyze:")
if st.button("Analyze"):
    if text:
        dist = model.transform([text])[0]
        st.success(f"Dominant Topic: {dist.argmax()}")
        st.bar_chart(dist)
