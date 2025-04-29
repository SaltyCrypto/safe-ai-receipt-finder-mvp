import streamlit as st
import pandas as pd
import openai

st.set_page_config(page_title="🧠 Safe AI Receipt Finder - Creative Scoring MVP", layout="wide")
st.title("🧠 Safe AI Receipt Finder - Creative Scoring MVP")

st.markdown("Upload your **ad hooks** or **user quotes** file below, and we'll let you preview, embed, and prep it for analysis.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

# API Key Input
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Function to call OpenAI Embeddings
def embed_text(text, key):
    openai.api_key = key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

if uploaded_file is not None and api_key:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔍 Preview of Your Data", df.head())

    if st.button("🧠 Embed All Hooks"):
        with st.spinner("Embedding in progress..."):
            df['embedding'] = df['Text'].apply(lambda x: embed_text(x, api_key))
        st.success("✅ Embedding complete!")
        st.write("### 📈 Preview Embedded Data", df.head())

        st.download_button(
            label="📥 Download CSV with Embeddings",
            data=df.to_csv(index=False),
            file_name="embedded_creatives.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Prototype powered by OpenAI + Streamlit • Built for creative scoring exploration.")
