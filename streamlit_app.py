import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity

# === Config ===
st.set_page_config(page_title="🧠 Safe AI Receipt Finder PRO", layout="wide")
st.title("🧠 Safe AI Receipt Finder – PRO Creative Analyzer")

# === API Key ===
api_key = st.sidebar.text_input("🔐 Enter your OpenAI API Key", type="password")
if not api_key or not api_key.startswith("sk-"):
    st.sidebar.warning("Enter a valid API key to begin.")
    st.stop()
openai.api_key = api_key

# === Session Setup ===
if "embedded_df" not in st.session_state:
    st.session_state.embedded_df = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "1️⃣ Embed"

# === UX Progress Wizard ===
st.markdown("#### 🚀 Workflow Progress")
st.markdown("""
✅ **Step 1:** Embed Creatives  
⬜ **Step 2:** Pick Diverse Creatives  
⬜ **Step 3:** Score Creatives  
⬜ **Step 4:** PRO Rewrites  
⬜ **Step 5:** A/B Picker  
⬜ **Step 6:** Snapshot + Exports  
""")

# === Tabs ===
tabs = st.tabs([
    "1️⃣ Embed", "2️⃣ Diversity Picker", "3️⃣ Creative Scorer",
    "4️⃣ PRO Rewrites", "5️⃣ A/B Picker", "6️⃣ Snapshot",
    "7️⃣ Emotion Lens", "8️⃣ Clustering"
])

# === TAB 1: Embed ===
with tabs[0]:
    st.subheader("📥 Upload & Embed")
    uploaded_file = st.file_uploader("Upload a CSV with a 'Text' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Text' not in df.columns:
            st.error("❌ Your file must contain a 'Text' column.")
        else:
            if st.button("🧠 Embed All Hooks"):
                st.info("Embedding up to 200 rows using `text-embedding-ada-002`")
                embeddings = []
                with st.spinner("Generating embeddings..."):
                    for i, row in df.iterrows():
                        if i >= 200:
                            break
                        try:
                            text = str(row['Text'])
                            response = openai.Embedding.create(
                                input=text,
                                model="text-embedding-ada-002"
                            )
                            embeddings.append(response['data'][0]['embedding'])
                        except Exception as e:
                            embeddings.append([0.0] * 1536)  # fallback
                        time.sleep(0.3)
                df['embedding'] = embeddings
                st.session_state.embedded_df = df
                st.success("✅ Embedding complete! Data saved to session.")
                st.dataframe(df[['Text']])
                st.button("➡️ Continue to Diversity Picker", on_click=lambda: st.session_state.update({"active_tab": "2️⃣ Diversity Picker"}))

# === TAB 2 to 8 Placeholder Logic ===
for i, label in enumerate([
    "Diversity Picker", "Creative Scorer", "PRO Rewrites",
    "A/B Picker", "Snapshot", "Emotion Lens", "Clustering"
], start=1):
    with tabs[i]:
        st.subheader(f"🛠️ {label}")
        if st.session_state.embedded_df is None:
            st.warning("⚠️ Please upload and embed data first in Tab 1.")
        else:
            st.info("🔧 Feature logic will go here...")

