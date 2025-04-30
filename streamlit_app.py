import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity

# === SETUP ===
st.set_page_config(page_title="🧠 Creative Scoring MVP", layout="centered")
st.title("🧠 Safe AI Receipt Finder — Creative Intelligence MVP")

tab1, tab2 = st.tabs(["📤 Embed + Score", "🎯 Pick Most Diverse"])

# === API Key ===
api_key = st.text_input("🔑 Paste your OpenAI API key", type="password")
client = None
if api_key:
    if api_key.startswith("sk-") and len(api_key) > 20:
        client = openai.OpenAI(api_key=api_key)
    else:
        st.error("⚠️ Invalid API key format")

# === SAFE COSINE SCORING ===
def safe_cosine_score(embedding_str, target_vec):
    try:
        vec = np.fromstring(embedding_str.strip('[]'), sep=',')
        if vec.shape[0] != target_vec.shape[0]:
            return "❌ Shape mismatch"
        if not np.any(vec):
            return "❌ Zero vector"
        return float(cosine_similarity(vec.reshape(1, -1), target_vec.reshape(1, -1))[0][0])
    except Exception as e:
        return f"❌ Error"

# === TAB 1: EMBED + SCORE ===
with tab1:
    uploaded_file = st.file_uploader("📁 Upload your CSV (must have 'Text' column)", type=["csv"])

    if uploaded_file and client:
        df = pd.read_csv(uploaded_file)

        if 'Text' not in df.columns:
            st.error("🚫 CSV must include a 'Text' column")
        elif len(df) > 200:
            st.error("🚫 Limit: 200 rows max")
        else:
            if st.button("🧠 Embed Texts"):
                with st.spinner("Generating embeddings..."):
                    embeddings = []
                    progress = st.progress(0)
                    for i, row in df.iterrows():
                        text = str(row['Text'])
                        try:
                            response = client.embeddings.create(
                                input=text,
                                model="text-embedding-ada-002"
                            )
                            embeddings.append(response.data[0].embedding)
                        except Exception:
                            embeddings.append(None)
                        progress.progress((i+1)/len(df))
                        time.sleep(0.4)

                    df['embedding'] = embeddings
                    st.success("✅ Embedding complete")

            # Scoring
            score_mode = st.selectbox("🎯 Target emotion?", ['Excitement', 'Urgency', 'Security'])
            st.write("Click to score against your chosen emotion.")

            if st.button("📊 Score Emotion"):
                try:
                    target_vec = client.embeddings.create(
                        input=score_mode,
                        model="text-embedding-ada-002"
                    ).data[0].embedding

                    df['emotion_score'] = df['embedding'].apply(lambda x: safe_cosine_score(str(x), np.array(target_vec)))
                    st.success("✅ Scoring complete")
                    st.dataframe(df[['Text', 'emotion_score']])
                except Exception as e:
                    st.error("❌ Error scoring: Check embeddings or API limits")

            if 'emotion_score' in df.columns:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download Results", data=csv, file_name="scored_creatives.csv", mime="text/csv")

    elif uploaded_file and not api_key:
        st.warning("🔑 Paste your API key to enable embedding")

# === TAB 2: DIVERSITY PICKER ===
with tab2:
    uploaded_diverse = st.file_uploader("📁 Upload Embedded CSV", type=["csv"], key="diverse")

    if uploaded_diverse:
        df = pd.read_csv(uploaded_diverse)

        if 'embedding' not in df.columns:
            st.error("🚫 'embedding' column required")
        else:
            try:
                df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(str(x).strip('[]'), sep=','))
            except Exception as e:
                st.error(f"Embedding parse error: {e}")

            k = st.slider("How many to pick?", min_value=2, max_value=min(10, len(df)), value=5)
            if st.button("🎯 Pick Most Diverse"):
                selected = [0]
                matrix = np.vstack(df['embedding'].to_numpy())

                while len(selected) < k:
                    remaining = list(set(range(len(df))) - set(selected))
                    scores = []
                    for idx in remaining:
                        sim = cosine_similarity(matrix[idx].reshape(1, -1), matrix[selected]).mean()
                        scores.append((idx, sim))
                    next_idx = min(scores, key=lambda x: x[1])[0]
                    selected.append(next_idx)

                diverse_df = df.iloc[selected][['Text']]
                st.dataframe(diverse_df)
                csv = diverse_df.to_csv(index=False)
                st.download_button("📥 Download Diverse Hooks", data=csv, file_name="diverse_creatives.csv", mime="text/csv")
