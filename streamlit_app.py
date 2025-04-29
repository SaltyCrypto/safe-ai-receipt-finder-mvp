# 📄 safe-ai-receipt-finder-mvp/streamlit_app.py (Final Bulletproof Version)

import streamlit as st
import pandas as pd
import openai
import time

# Set page config
st.set_page_config(page_title="🧠 Safe AI Receipt Finder - Creative Scoring MVP", layout="centered")

# Title
st.title("🧠 Safe AI Receipt Finder - Creative Scoring MVP")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file (Max 200 rows)", type=["csv"])

# API key input
api_key = st.text_input("Paste your OpenAI API key", type="password")

# Validate API key format
def is_valid_api_key(key):
    return key.startswith("sk-") and len(key) > 20

# Initialize OpenAI client if API key provided and valid
client = None
if api_key:
    if is_valid_api_key(api_key):
        client = openai.OpenAI(api_key=api_key)
    else:
        st.error("Invalid API key format. Please check and try again.")

# Main logic
if uploaded_file and client:
    df = pd.read_csv(uploaded_file)

    if len(df) > 200:
        st.error("🚫 Upload limited to 200 rows maximum. Please upload a smaller file.")
    elif 'Text' not in df.columns:
        st.error("Your CSV must have a 'Text' column.")
    else:
        if st.button("🧠 Embed All Hooks"):
            with st.spinner('Embedding texts...'):
                progress_bar = st.progress(0)
                embeddings = []
                total = len(df)

                for idx, text in enumerate(df['Text']):
                    if pd.isna(text) or str(text).strip() == '':
                        embeddings.append(None)
                        progress_bar.progress((idx + 1) / total)
                        continue

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
                            time.sleep(5)
                        except Exception as e:
                            retries -= 1
                            if retries == 0:
                                st.error(f"Failed to embed text at row {idx} after retries. Skipping.")
                                embeddings.append(None)
                                success = True
                    progress_bar.progress((idx + 1) / total)
                    time.sleep(1)

                if len(embeddings) == len(df):
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
                else:
                    st.error("⚠️ Internal mismatch detected. Please try re-uploading.")

elif uploaded_file and not api_key:
    st.warning("🔑 Please enter your OpenAI API key to proceed.")

else:
    st.info("📤 Upload a CSV and paste your API key to start.")
