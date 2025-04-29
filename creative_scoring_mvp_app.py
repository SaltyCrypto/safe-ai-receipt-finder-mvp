import streamlit as st
import pandas as pd
import openai

# Set Streamlit page config
st.set_page_config(page_title="🧾 Safe AI Receipt Finder - Creative Scoring MVP", layout="wide")
st.title("🧾 Safe AI Receipt Finder - Creative Scoring MVP")

st.markdown("""
Upload your **hooks or user quotes** CSV, 
enter your **OpenAI API Key**, 
and embed your creatives for smarter clustering and scoring.
""")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

# API Key input (masked)
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Function to embed text
def embed_text(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Main app logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔍 Preview of Your Data", df.head())

    if "Text" not in df.columns:
        st.error("❌ Your CSV must contain a column named 'Text'. Please fix and upload again.")
    else:
        if st.button("🧠 Embed All Hooks"):
            if api_key:
                openai.api_key = api_key
                with st.spinner("🔄 Embedding in progress... Please wait."):
                    df['embedding'] = df['Text'].apply(embed_text)
                st.success("✅ Embedding complete!")

                st.write("### 📈 Preview of Embedded Data", df.head())

                st.download_button(
                    label="📥 Download CSV with Embeddings",
                    data=df.to_csv(index=False),
                    file_name="embedded_creatives.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ Please enter your OpenAI API key to proceed.")

st.markdown("---")
st.caption("Built by 🚀 - Safe AI Receipt Finder Project")
