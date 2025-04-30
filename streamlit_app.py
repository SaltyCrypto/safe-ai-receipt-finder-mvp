
# streamlit_app.py — Safe AI Receipt Finder PRO v5 with Google Ads KW Embed (Tab 3)

import streamlit as st
import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="🧠 Safe AI Receipt Finder - PRO Creative Intelligence OS v5", layout="wide")

# Sidebar for API Key
api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
client = None
if api_key and api_key.startswith("sk-"):
    client = openai.OpenAI(api_key=api_key)
else:
    st.sidebar.warning("Please provide a valid OpenAI API Key")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🪄 Hook Generator", "🧠 Emotion Lens", "📊 KW Embedder", 
    "🎯 Diversity Picker", "📈 Creative Scoring", "⚖️ A/B Simulator", "📸 Snapshot Export"
])

# Tab 3: KW Embedder
with tab3:
    st.subheader("📊 Google Keyword Embedding")

    uploaded_kw_file = st.file_uploader("Upload CSV with Keywords + Volume", type=["csv"], key="kw")

    if uploaded_kw_file and client:
        kw_df = pd.read_csv(uploaded_kw_file)
        if 'keyword' not in kw_df.columns or 'volume' not in kw_df.columns:
            st.error("CSV must include 'keyword' and 'volume' columns.")
        else:
            if st.button("🔍 Embed Keywords"):
                with st.spinner("Embedding keywords..."):
                    progress = st.progress(0)
                    kw_embeddings = []
                    for i, text in enumerate(kw_df['keyword']):
                        try:
                            res = client.embeddings.create(input=text, model="text-embedding-ada-002")
                            kw_embeddings.append(res.data[0].embedding)
                        except:
                            kw_embeddings.append([0]*1536)
                        progress.progress((i+1)/len(kw_df))
                        time.sleep(0.25)

                    kw_df['embedding'] = kw_embeddings
                    st.success("✅ Embedded keywords successfully!")
                    st.dataframe(kw_df[['keyword', 'volume']])
                    st.download_button("📥 Download Embedded Keywords CSV", kw_df.to_csv(index=False), file_name="embedded_keywords.csv")
