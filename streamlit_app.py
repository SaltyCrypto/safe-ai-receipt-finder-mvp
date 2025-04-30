import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🧠 Safe AI Receipt Finder - PRO Creative Analyzer", layout="wide")

st.title("🧠 Safe AI Receipt Finder - PRO Creative Analyzer")

tabs = st.tabs([
    "1️⃣ Embed",
    "2️⃣ Diversity Picker",
    "3️⃣ Creative Scorer",
    "4️⃣ PRO Rewrites",
    "5️⃣ A/B Picker",
    "6️⃣ Intelligence Profile"
])

# --- Helper: Validate API key ---
def is_valid_api_key(key):
    return key.startswith("sk-") and len(key) > 20

# --- Helper: Embed Text ---
def embed_text_list(texts, client):
    embeddings = []
    for i, text in enumerate(texts):
        if pd.isna(text) or str(text).strip() == '':
            embeddings.append(None)
            continue
        try:
            response = client.embeddings.create(input=text, model="text-embedding-ada-002")
            embeddings.append(response.data[0].embedding)
        except Exception:
            embeddings.append(None)
        time.sleep(0.5)
    return embeddings

# --- Tab 1: Embed ---
with tabs[0]:
    st.header("📌 Step 1: Embed Creatives")
    api_key = st.text_input("🔑 Paste your OpenAI API Key", type="password", key="api1")
    uploaded = st.file_uploader("📤 Upload your CSV with a 'Text' column", type=["csv"], key="embed")

    if uploaded and api_key and is_valid_api_key(api_key):
        client = openai.OpenAI(api_key=api_key)
        df = pd.read_csv(uploaded)

        if 'Text' not in df.columns:
            st.error("Missing 'Text' column.")
        else:
            if st.button("🧠 Embed Now"):
                with st.spinner("Generating embeddings..."):
                    df['embedding'] = embed_text_list(df['Text'].tolist(), client)
                    st.success("✅ Done!")
                    st.dataframe(df.head())
                    st.download_button("📥 Download", df.to_csv(index=False), "embedded.csv")

# --- Tab 2: Diversity Picker ---
with tabs[1]:
    st.header("🎯 Pick Most Diverse Creatives")
    uploaded_diverse = st.file_uploader("📤 Upload your Embedded CSV", type=["csv"], key="diverse")
    if uploaded_diverse:
        df = pd.read_csv(uploaded_diverse)
        if 'embedding' not in df.columns:
            st.error("Missing 'embedding' column.")
        else:
            df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
            k = st.slider("How many diverse creatives?", 2, min(10, len(df)), 5)
            if st.button("🚀 Pick Diverse"):
                selected = [0]
                emb = np.vstack(df['embedding'])
                while len(selected) < k:
                    remaining = list(set(range(len(df))) - set(selected))
                    scores = [(i, cosine_similarity([emb[i]], emb[selected]).mean()) for i in remaining]
                    next_idx = min(scores, key=lambda x: x[1])[0]
                    selected.append(next_idx)
                diverse_df = df.iloc[selected][['Text']]
                st.dataframe(diverse_df)
                st.download_button("📥 Download Diverse", diverse_df.to_csv(index=False), "diverse_creatives.csv")

# --- Tab 3: Creative Scorer ---
with tabs[2]:
    st.header("🔢 Score Creatives")
    uploaded_score = st.file_uploader("📤 Upload your Embedded CSV", type=["csv"], key="scorer")
    score_type = st.selectbox("🎯 Select scoring model", ["🧠 Emotion Strength", "🧲 Persuasion Score", "💰 Monetization Angle"])
    prompt_templates = {
        "🧠 Emotion Strength": "How emotionally impactful is this message?",
        "🧲 Persuasion Score": "Rate the persuasive power of this ad hook.",
        "💰 Monetization Angle": "Estimate the monetization potential of this creative."
    }

    if uploaded_score and api_key and is_valid_api_key(api_key):
        client = openai.OpenAI(api_key=api_key)
        df = pd.read_csv(uploaded_score)
        if 'Text' not in df.columns:
            st.error("Missing 'Text' column.")
        else:
            if st.button("⚖️ Score Creatives"):
                scores = []
                for text in df['Text']:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": prompt_templates[score_type]},
                                {"role": "user", "content": text}
                            ]
                        )
                        score_str = response.choices[0].message.content
                        score = int(''.join(filter(str.isdigit, score_str)))
                        scores.append(score)
                    except:
                        scores.append(0)
                    time.sleep(0.5)
                df[score_type] = scores
                st.success("✅ Scoring complete!")
                st.dataframe(df.head())
                st.download_button("📥 Download Scored", df.to_csv(index=False), "scored_creatives.csv")

# --- Tab 4: PRO Rewrites ---
with tabs[3]:
    st.header("✍️ Rewrite Creatives PRO Mode")
    uploaded_rewrite = st.file_uploader("📤 Upload your CSV with 'Text'", type=["csv"], key="rewrite")
    rewrite_prompt = st.text_area("🧠 Enter rewrite prompt", "Rewrite this to be more urgent, clear, and emotional.")

    if uploaded_rewrite and api_key and is_valid_api_key(api_key):
        client = openai.OpenAI(api_key=api_key)
        df = pd.read_csv(uploaded_rewrite)
        if 'Text' not in df.columns:
            st.error("Missing 'Text' column.")
        else:
            if st.button("✍️ Rewrite Now"):
                rewrites = []
                for text in df['Text']:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": rewrite_prompt},
                                {"role": "user", "content": text}
                            ]
                        )
                        rewrites.append(response.choices[0].message.content.strip())
                    except:
                        rewrites.append(text)
                    time.sleep(0.5)
                df['Rewrite'] = rewrites
                st.dataframe(df.head())
                st.download_button("📥 Download Rewrites", df.to_csv(index=False), "rewritten_creatives.csv")

# --- Tab 5: A/B Picker ---
with tabs[4]:
    st.header("🔀 A/B Test Picker")
    uploaded_ab = st.file_uploader("📤 Upload your Scored CSV", type=["csv"], key="ab")
    if uploaded_ab:
        df = pd.read_csv(uploaded_ab)
        st.markdown("📊 Pick 2 creatives for A/B testing")
        indices = st.multiselect("Select rows", df.index.tolist(), default=df.index[:2].tolist())
        if len(indices) == 2:
            ab_df = df.iloc[indices]
            st.dataframe(ab_df)
            st.download_button("📥 Download A/B Pair", ab_df.to_csv(index=False), "ab_test_pair.csv")
        else:
            st.warning("Please select exactly 2 rows.")

# --- Tab 6: Intelligence Profile ---
with tabs[5]:
    st.header("📊 Creative Intelligence Snapshot")
    uploaded_intel = st.file_uploader("📤 Upload Final Scored CSV", type=["csv"], key="intel")
    if uploaded_intel:
        df = pd.read_csv(uploaded_intel)
        if 'Text' not in df.columns:
            st.error("Missing 'Text' column.")
        else:
            st.markdown("### 📌 Snapshot")
            st.write("Rows:", len(df))
            st.write("Columns:", df.columns.tolist())
            st.dataframe(df.head())