# 📄 safe-ai-receipt-finder-mvp/streamlit_app.py (PRO Mode)

import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity

# 🚀 Set up page
st.set_page_config(page_title="🧠 Safe AI Receipt Finder PRO", layout="centered")
st.title("🧠 Safe AI Receipt Finder - PRO Creative Analyzer")

# 🚀 File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV (must have 'Text' column)", type=["csv"])

# 🚀 API Key input
api_key = st.text_input("🔑 Paste your OpenAI API key here", type="password")

# ✅ Validate key
def is_valid_api_key(key):
    return key.startswith("sk-") and len(key) > 20

client = None
if api_key:
    if is_valid_api_key(api_key):
        client = openai.OpenAI(api_key=api_key)
    else:
        st.error("🚫 Invalid API key format!")

# ✅ Auto-frame function
def auto_frame(text):
    return f"Creative marketing hook. Goal: drive action. Emotion focus. Text: {text}"

# ✅ Emotion prediction function
def predict_emotion(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a marketing psychology expert. Choose ONE main emotion: fear, hope, greed, excitement, sadness, curiosity, envy, pride, anger, love, other."},
                {"role": "user", "content": f"What is the main emotion triggered by this creative? '{text}'"}
            ],
            temperature=0.2,
            max_tokens=10,
        )
        emotion = response.choices[0].message.content.strip()
        return emotion
    except Exception as e:
        return "Unknown"

# 🚀 Main logic
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

                    framed = auto_frame(raw_text)
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
                    emotion = predict_emotion(raw_text)
                    emotions.append(emotion)

                    progress.progress((idx + 1) / len(df))
                    time.sleep(0.3)  # slight pacing

                # Assign results
                df['Framed Text'] = framed_texts
                df['Embedding'] = embeddings
                df['Predicted Emotion'] = emotions

                st.success("✅ Processing complete!")
                st.dataframe(df)

                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full PRO CSV",
                    data=csv,
                    file_name='pro_creative_analysis.csv',
                    mime='text/csv'
                )

elif uploaded_file and not api_key:
    st.warning("🔑 Please enter your OpenAI API key to continue.")

else:
    st.info("📤 Upload a CSV and enter your OpenAI API key to begin.")

