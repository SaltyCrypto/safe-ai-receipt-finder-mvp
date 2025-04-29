import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🧠 Safe AI Receipt Finder - Creative MVP", layout="centered")

st.title("🧠 Safe AI Receipt Finder - Creative Scoring & Diversity Picker")

# Create Tabs
tab1, tab2 = st.tabs(["🔹 Embed Creatives", "🔹 Pick Diverse Creatives"])

# ==== TAB 1: Embedding ====
with tab1:
    st.header("📤 Upload and Embed Creatives")
    uploaded_file = st.file_uploader("Upload your CSV file (Max 200 rows)", type=["csv"], key="embed")

    # API key input
    api_key = st.text_input("Paste your OpenAI API key", type="password")

    def is_valid_api_key(key):
        return key.startswith("sk-") and len(key) > 20

    client = None
    if api_key:
        if is_valid_api_key(api_key):
            client = openai.OpenAI(api_key=api_key)
        else:
            st.error("Invalid API key format. Please check and try again.")

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
                        else:
                            try:
                                response = client.embeddings.create(
                                    input=text,
                                    model="text-embedding-ada-002"
                                )
                                embeddings.append(response.data[0].embedding)
                            except Exception as e:
                                embeddings.append(None)
                        progress_bar.progress((idx + 1) / total)
                        time.sleep(0.4)

                    if len(embeddings) == len(df):
                        df['embedding'] = embeddings
                        st.success("✅ Embedding complete!")
                        st.dataframe(df.head())

                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Embedded CSV",
                            data=csv,
                            file_name='embedded_creatives.csv',
                            mime='text/csv'
                        )
                    else:
                        st.error("❌ Major mismatch detected. Please re-upload and retry.")

    elif uploaded_file and not api_key:
        st.warning("🔑 Please enter your OpenAI API key to proceed.")

    else:
        st.info("📤 Upload a CSV and paste your API key to start.")

# ==== TAB 2: Diversity Picker ====
with tab2:
    st.header("🎯 Pick Most Diverse Creatives")

    uploaded_diverse = st.file_uploader("📤 Upload your Embedded CSV", type=["csv"], key="diverse")

    if uploaded_diverse:
        df = pd.read_csv(uploaded_diverse)

        if 'embedding' not in df.columns:
            st.error("🚫 The CSV must have an 'embedding' column!")
        else:
            try:
                df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
            except Exception as e:
                st.error(f"Embedding parsing error: {e}")

            k = st.slider('How many diverse creatives?', 2, min(10, len(df)), 5, key="slider_diverse")

            if st.button("🚀 Pick Diverse"):
                selected = [0]
                embeddings = np.vstack(df['embedding'].to_numpy())

                while len(selected) < k:
                    remaining = list(set(range(len(df))) - set(selected))
                    scores = []
                    for idx in remaining:
                        similarity = cosine_similarity(embeddings[idx].reshape(1, -1), embeddings[selected]).mean()
                        scores.append((idx, similarity))
                    next_idx = min(scores, key=lambda x: x[1])[0]
                    selected.append(next_idx)

                diverse_df = df.iloc[selected][['Text']]
                st.success(f"🎯 Picked {k} diverse creatives!")
                st.dataframe(diverse_df)

                csv = diverse_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Diverse Creatives",
                    data=csv,
                    file_name='diverse_creatives.csv',
                    mime='text/csv'
                )
