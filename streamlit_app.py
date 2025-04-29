# 📄 safe-ai-receipt-finder-mvp/streamlit_app.py (Final Clean Version)

import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity

# 🚀 Set up page
st.set_page_config(page_title="🧠 Safe AI Receipt Finder PRO", layout="centered")
st.title("🧠 Safe AI Receipt Finder - PRO Creative Analyzer")

# 📤 File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV (must have 'Text' column)", type=["csv"])

# 🔑 API Key input
api_key = st.text_input("🔑 Paste your OpenAI API key", type="password")

# ✅ Validate API Key
def is_valid_api_key(key):
    return key.startswith("sk-") and len(key) > 20

client = None
if api_key:
    if is_valid_api_key(api_key):
        client = openai.OpenAI(api_key=api_key)
    else:
        st.error("🚫 Invalid API key format!")

# 📑 Setup Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Embed Creatives", "🎯 Pick Diverse Creatives", "📈 Score by Emotion"])

# 🔥 TAB 1: Embed Creatives
with tab1:
    st.header("🔍 Embed Creatives + Predict Emotion")

    if uploaded_file and client:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin1")

        if 'Text' not in df.columns:
            st.error("🚫 Your CSV must have a 'Text' column.")
        else:
            if st.button("🚀 Process Creatives"):
                with st.spinner('Working... 🚀'):
                    progress = st.progress(0)

                    framed_texts = []
                    embeddings = []
                    emotions = []

                    for idx, row in df.iterrows():
                        raw_text = str(row['Text'])
                        if not raw_text.strip():
                            framed_texts.append("")
                            embeddings.append(None)
                            emotions.append("Unknown")
                            continue

                        framed = f"Creative marketing hook. Goal: drive action. Emotion focus. Text: {raw_text}"
                        framed_texts.append(framed)

                        # Embed
                        try:
                            response = client.embeddings.create(
                                input=framed,
                                model="text-embedding-ada-002"
                            )
                            embedding = response.data[0].embedding
                            embeddings.append(embedding)
                        except Exception as e:
                            embeddings.append(None)

                        # Predict Emotion
                        try:
                            response_emotion = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a marketing psychology expert. Choose ONE main emotion: fear, hope, greed, excitement, sadness, curiosity, envy, pride, anger, love, other."},
                                    {"role": "user", "content": f"What is the main emotion triggered by this creative? '{raw_text}'"}
                                ],
                                temperature=0.2,
                                max_tokens=10,
                            )
                            emotion = response_emotion.choices[0].message.content.strip()
                            emotions.append(emotion)
                        except Exception as e:
                            emotions.append("Unknown")

                        progress.progress((idx + 1) / len(df))
                        time.sleep(0.3)

                    df['Framed Text'] = framed_texts
                    df['Embedding'] = embeddings
                    df['Predicted Emotion'] = emotions

                    st.success("✅ Processing complete!")
                    st.dataframe(df)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Embedded CSV",
                        data=csv,
                        file_name='pro_creative_analysis.csv',
                        mime='text/csv'
                    )

    elif uploaded_file and not api_key:
        st.warning("🔑 Please enter your OpenAI API key to continue.")

# 🔥 TAB 2: Pick Diverse Creatives
with tab2:
    st.header("🎯 Pick Most Diverse Creatives")

    uploaded_diverse = st.file_uploader("📤 Upload Embedded CSV", type=["csv"], key="diverse")

    if uploaded_diverse:
        df = pd.read_csv(uploaded_diverse)

        if 'Embedding' not in df.columns:
            st.error("🚫 CSV must have 'Embedding' column!")
        else:
            try:
                df['Embedding'] = df['Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
            except Exception as e:
                st.error(f"Embedding parsing error: {e}")

            k = st.slider('How many diverse creatives?', 2, min(10, len(df)), 5, key="slider_diverse")

            if st.button("🚀 Pick Diverse"):
                selected = [0]
                embeddings = np.vstack(df['Embedding'].to_numpy())

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

# 🔥 TAB 3: Score by Emotion
with tab3:
    st.header("📈 Rank Creatives by Target Emotion")

    uploaded_score = st.file_uploader("📤 Upload Embedded CSV with Emotion", type=["csv"], key="score_emotion")

    if uploaded_score:
        df = pd.read_csv(uploaded_score)

        if 'Predicted Emotion' not in df.columns or 'Text' not in df.columns:
            st.error("🚫 CSV must have 'Predicted Emotion' and 'Text' columns!")
        else:
            emotions = ['Fear', 'Hope', 'Curiosity', 'Love', 'Greed', 'Excitement', 'Pride', 'Anger', 'Envy', 'Other']
            target = st.selectbox("🎯 Choose Target Emotion", emotions)

            def score_match(predicted, target):
                if pd.isna(predicted): return 0.0
                predicted = predicted.strip().lower()
                target = target.strip().lower()
                if predicted == target:
                    return 1.0
                elif target in predicted or predicted in target:
                    return 0.5
                else:
                    return 0.0

            df['Emotion Score'] = df['Predicted Emotion'].apply(lambda e: score_match(e, target))
            df_sorted = df.sort_values(by='Emotion Score', ascending=False)

            st.success(f"🎯 Ranked by how closely creatives match: **{target}**")
            st.dataframe(df_sorted[['Text', 'Predicted Emotion', 'Emotion Score']].head(20))

            csv = df_sorted.to_csv(index=False)
            st.download_button(
                label="📥 Download Ranked Creatives",
                data=csv,
                file_name=f'ranked_creatives_{target.lower()}.csv',
                mime='text/csv'
            )
