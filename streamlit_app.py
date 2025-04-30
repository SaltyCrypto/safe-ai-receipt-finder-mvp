
# 🧠 Safe AI Receipt Finder – PRO Creative Analyzer (All Tabs)

import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🧠 Safe AI Receipt Finder – PRO", layout="wide")
st.title("🧠 Safe AI Receipt Finder – PRO Creative Analyzer")

# Session state
if "client" not in st.session_state:
    st.session_state.client = None
if "df" not in st.session_state:
    st.session_state.df = None

# --- API KEY ---
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
if api_key and api_key.startswith("sk-"):
    try:
        openai_client = openai.OpenAI(api_key=api_key)
        st.session_state.client = openai_client
    except:
        st.sidebar.error("❌ Invalid API key")
else:
    st.sidebar.warning("Enter your API key")

# --- TABS ---
tabs = st.tabs([
    "1️⃣ Upload & Embed",
    "2️⃣ Hook Generator",
    "3️⃣ Diversity Picker",
    "4️⃣ Creative Scoring",
    "5️⃣ Emotion Lens",
    "6️⃣ A/B Simulator",
    "📸 Snapshot"
])

# --- TAB 1: Upload & Embed ---
with tabs[0]:
    st.header("📤 Upload & Embed Creatives")
    uploaded_file = st.file_uploader("Upload CSV with 'Text' column", type="csv", key="upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Text' not in df.columns:
            st.error("CSV must contain a 'Text' column")
        elif st.session_state.client:
            if st.button("🧠 Embed Now"):
                with st.spinner("Generating embeddings..."):
                    embeddings = []
                    for i, text in enumerate(df['Text']):
                        try:
                            if pd.isna(text) or not str(text).strip():
                                embeddings.append(None)
                            else:
                                res = st.session_state.client.embeddings.create(
                                    input=text,
                                    model="text-embedding-ada-002"
                                )
                                embeddings.append(res.data[0].embedding)
                        except Exception as e:
                            embeddings.append(None)
                        time.sleep(0.2)
                    df['embedding'] = embeddings
                    st.session_state.df = df
                    st.success("✅ Done!")
                    st.dataframe(df.head())
        else:
            st.warning("Please enter your API key to proceed")

# --- TAB 2: Hook Generator ---
with tabs[1]:
    st.header("🧠 Generate Hooks with LLM")
    prompt_frame = st.selectbox("Pick a framing style", ["Fear of Loss", "Tax Refund Boost", "Organizational Relief"])
    base_idea = st.text_input("💡 Base Product/Idea")
    if st.button("🚀 Generate Hooks"):
        if st.session_state.client and base_idea:
            with st.spinner("Generating..."):
                full_prompt = f"Create 5 short, high-converting hooks for '{base_idea}' framed as: {prompt_frame}"
                try:
                    res = st.session_state.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    hooks = res.choices[0].message.content
                    st.text_area("✍️ Hooks", hooks, height=200)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Enter idea + API key")

# --- TAB 3: Diversity Picker ---
with tabs[2]:
    st.header("🎯 Diverse Creative Picker")
    diverse_file = st.file_uploader("Upload Embedded CSV", type=["csv"], key="diverse")
    if diverse_file:
        df = pd.read_csv(diverse_file)
        try:
            df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
            k = st.slider("How many?", 2, 10, 5)
            if st.button("Pick"):
                selected = [0]
                matrix = np.vstack(df['embedding'])
                while len(selected) < k:
                    remaining = list(set(range(len(matrix))) - set(selected))
                    scores = [(i, cosine_similarity(matrix[i].reshape(1, -1), matrix[selected]).mean()) for i in remaining]
                    next_i = min(scores, key=lambda x: x[1])[0]
                    selected.append(next_i)
                diverse_df = df.iloc[selected]
                st.success("✅ Selected")
                st.dataframe(diverse_df[['Text']])
        except Exception as e:
            st.error("Embedding parse error")

# --- TAB 4: Scoring ---
with tabs[3]:
    st.header("📊 Score Creatives (LLM-based)")
    if st.session_state.df is not None:
        score_prompt = st.text_area("Prompt for scoring (e.g., 'Rate based on urgency appeal')")
        if st.button("Score Creatives"):
            with st.spinner("Scoring..."):
                scores = []
                for i, row in st.session_state.df.iterrows():
                    try:
                        msg = [{"role": "user", "content": f"Rate this creative (1-100): {row['Text']}\nPrompt: {score_prompt}"}]
                        res = st.session_state.client.chat.completions.create(model="gpt-4", messages=msg)
                        score = int(''.join(filter(str.isdigit, res.choices[0].message.content.split()[0])))
                        scores.append(score)
                    except:
                        scores.append(0)
                st.session_state.df['Score'] = scores
                st.dataframe(st.session_state.df[['Text', 'Score']])
    else:
        st.warning("Upload & embed first.")

# --- TAB 5: Emotion Lens ---
with tabs[4]:
    st.header("🧠 Emotion Detection & Lens")
    if st.session_state.df is not None:
        if st.button("🔍 Detect Emotions"):
            emotions = []
            for text in st.session_state.df['Text']:
                try:
                    res = st.session_state.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": f"What emotion does this hook evoke? '{text}'? Just the emotion word."}]
                    )
                    emotions.append(res.choices[0].message.content.strip())
                except:
                    emotions.append("Unknown")
            st.session_state.df['Emotion'] = emotions
            st.dataframe(st.session_state.df[['Text', 'Emotion']])
    else:
        st.warning("Upload & embed first.")

# --- TAB 6: A/B Simulator ---
with tabs[5]:
    st.header("🧪 A/B Test Simulator")
    if st.session_state.df is not None:
        idx1 = st.selectbox("Creative A", st.session_state.df.index, format_func=lambda i: st.session_state.df.loc[i, 'Text'])
        idx2 = st.selectbox("Creative B", st.session_state.df.index, format_func=lambda i: st.session_state.df.loc[i, 'Text'])
        if st.button("🤖 Pick Winner"):
            try:
                a = st.session_state.df.loc[idx1, 'Text']
                b = st.session_state.df.loc[idx2, 'Text']
                msg = [{"role": "user", "content": f"Between these, which will likely perform better in an ad?\nA: {a}\nB: {b}"}]
                res = st.session_state.client.chat.completions.create(model="gpt-4", messages=msg)
                st.success(res.choices[0].message.content.strip())
            except:
                st.error("Something went wrong")
    else:
        st.warning("Upload & embed first.")

# --- TAB 7: Snapshot ---
with tabs[6]:
    st.header("📸 Export Snapshot")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df)
        csv = st.session_state.df.to_csv(index=False)
        st.download_button("📥 Download Final CSV", csv, "creative_analysis_snapshot.csv", "text/csv")
    else:
        st.info("Nothing to export yet.")
