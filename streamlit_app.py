
from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Page config
st.set_page_config(page_title="🧠 Safe AI Receipt Finder – PRO Creative Intelligence OS", layout="wide")

st.title("🧠 Safe AI Receipt Finder – PRO Creative Intelligence OS")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Hook Generator",
    "2️⃣ Emotion Lens",
    "3️⃣ Diversity Picker",
    "4️⃣ Creative Scoring",
    "5️⃣ A/B Simulator",
    "6️⃣ Snapshot + Export"
])

# API key setup
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

# ===================== TAB 1: HOOK GENERATOR =====================
with tab1:
    st.header("✍️ Generate Hooks with Prompt Frames")

    prompt_frame = st.selectbox("Choose a Frame", ["Fear of Loss", "Monetary Gain", "Privacy", "Efficiency", "Other"])
    input_idea = st.text_area("Enter your product / concept")

    if st.button("Generate Hooks") and input_idea and client:
        prompt = f"Generate 5 high-converting ad hooks based on {prompt_frame} for this product:
{input_idea}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content
        st.success("Generated Hooks")
        st.text_area("Hooks", value=output, height=200)

# ===================== TAB 2: EMOTION LENS =====================
with tab2:
    st.header("🧠 Embed + Emotion Lens")

    uploaded_file = st.file_uploader("Upload CSV with 'Text' column", type=["csv"], key="embed_upload")
    if uploaded_file and client:
        df = pd.read_csv(uploaded_file)
        if 'Text' not in df.columns:
            st.error("CSV must contain a 'Text' column.")
        else:
            embeddings = []
            with st.spinner("Embedding texts..."):
                for text in df['Text']:
                    try:
                        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
                        embeddings.append(response.data[0].embedding)
                    except:
                        embeddings.append(None)
                    time.sleep(0.2)
            df['embedding'] = embeddings
            st.success("Embeddings complete!")
            st.dataframe(df.head())

            csv = df.to_csv(index=False)
            st.download_button("📥 Download with Embeddings", data=csv, file_name="embedded_hooks.csv", mime="text/csv")

# ===================== TAB 3: DIVERSITY PICKER =====================
with tab3:
    st.header("🎯 Pick Most Diverse Creatives")

    uploaded_diverse = st.file_uploader("Upload Embedded CSV", type=["csv"], key="diverse")
    if uploaded_diverse:
        df = pd.read_csv(uploaded_diverse)
        if 'embedding' in df.columns:
            try:
                df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
                k = st.slider("How many to pick?", 2, min(10, len(df)), 5)
                if st.button("Pick Diverse"):
                    selected = [0]
                    emb = np.vstack(df['embedding'].to_numpy())
                    while len(selected) < k:
                        rem = list(set(range(len(df))) - set(selected))
                        scores = [(i, cosine_similarity(emb[i].reshape(1, -1), emb[selected]).mean()) for i in rem]
                        selected.append(min(scores, key=lambda x: x[1])[0])
                    diverse_df = df.iloc[selected][['Text']]
                    st.dataframe(diverse_df)
                    st.download_button("📥 Download", data=diverse_df.to_csv(index=False), file_name="diverse.csv")
            except Exception as e:
                st.error(f"Error: {e}")

# ===================== TAB 4: CREATIVE SCORING =====================
with tab4:
    st.header("📊 Score Creatives")

    uploaded_score = st.file_uploader("Upload Embedded CSV", type=["csv"], key="score")
    if uploaded_score and client:
        df = pd.read_csv(uploaded_score)
        if 'embedding' in df.columns:
            df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
            if st.button("Score Creatives"):
                scores = []
                for text in df['Text']:
                    prompt = f"Rate this hook on a scale from 1–100 for persuasion, monetization potential, and emotional impact:
'{text}'"
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        score = response.choices[0].message.content
                        scores.append(score)
                    except:
                        scores.append("Error")
                df['LLM_Score'] = scores
                st.dataframe(df)
                st.download_button("📥 Download Scored", data=df.to_csv(index=False), file_name="scored_creatives.csv")

# ===================== TAB 5: A/B SIMULATOR =====================
with tab5:
    st.header("🧪 A/B Simulator")

    uploaded_ab = st.file_uploader("Upload Hooks", type=["csv"], key="ab")
    if uploaded_ab and client:
        df = pd.read_csv(uploaded_ab)
        if 'Text' in df.columns:
            hook1 = st.selectbox("Hook A", df['Text'].tolist(), index=0)
            hook2 = st.selectbox("Hook B", df['Text'].tolist(), index=1)
            if st.button("Run A/B Test"):
                prompt = f"Which of the two hooks below is more persuasive for a general audience?
A: {hook1}
B: {hook2}
Explain why."
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)

# ===================== TAB 6: SNAPSHOT + EXPORT =====================
with tab6:
    st.header("📸 Export Snapshot")
    st.info("Download final outputs or combine filtered views.")
