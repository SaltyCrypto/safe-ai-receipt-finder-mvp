# 📄 safe-ai-receipt-finder-mvp/streamlit_app.py (Pro Version)

import streamlit as st
import pandas as pd
import openai
import time

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
                progress_bar = st.progress(0)
                embeddings = []
                total = len(df)

                for idx, text in enumerate(df['Text']):
                    success = False
                    retries = 3
                    while not success and retries > 0:
                        try:
                            response = client.embeddings.create(
                                input=text,
                                model="text-embedding-ada-002"
                            )
                            embeddings.append(response.data[0].embedding)
                            success = True
                        except openai.RateLimitError:
                            retries -= 1
                            time.sleep(5)  # wait before retrying
                        except Exception as e:
                            st.error(f"Unexpected error: {e}")
                            embeddings.append(None)
                            success = True  # Skip to next after error
                    progress_bar.progress((idx + 1) / total)
                    time.sleep(1)  # gentle wait to avoid rate limits

                df['embedding'] = embeddings

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
