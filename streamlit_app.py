# 📄 safe-ai-receipt-finder-mvp/streamlit_app.py (Fresh Clean Version)

import streamlit as st
import pandas as pd
import openai

# Set page config
st.set_page_config(page_title="🧠 Safe AI Receipt Finder - Creative Scoring MVP", layout="centered")

# Title
st.title("🧠 Safe AI Receipt Finder - Creative Scoring MVP")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# API key input
api_key = st.text_input("Paste your OpenAI API key", type="password")

# Initialize OpenAI client if API key provided
client = None
if api_key:
    client = openai.OpenAI(api_key=api_key)

# Main logic
if uploaded_file and client:
    df = pd.read_csv(uploaded_file)

    if 'Text' not in df.columns:
        st.error("Your CSV must have a 'Text' column.")
    else:
        if st.button("🧠 Embed All Hooks"):
            with st.spinner('Embedding texts...'):
                # Embed all texts
                def embed_text(text):
                    response = client.embeddings.create(
                        input=text,
                        model="text-embedding-ada-002"
                    )
                    return response.data[0].embedding

                df['embedding'] = df['Text'].apply(embed_text)

                st.success("✅ Embedding complete!")
                st.dataframe(df.head())

                # Download CSV button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Embedded CSV",
                    data=csv,
                    file_name='embedded_creatives.csv',
                    mime='text/csv'
                )

elif uploaded_file and not api_key:
    st.warning("🔑 Please enter your OpenAI API key to proceed.")

else:
    st.info("📤 Upload a CSV and paste your API key to start.")
