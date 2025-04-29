
import streamlit as st
import pandas as pd
import openai

st.set_page_config(page_title="Creative Scoring MVP", layout="wide")

st.title("🧠 Safe AI Receipt Finder - Creative Scoring MVP")

st.markdown("""
Upload your **ad hooks** or **user quotes** file below, and we'll let you preview, filter, and prep it for embedding and scoring.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### Preview of Your Data", df.head())

    # Filter by source
    source_filter = st.selectbox("Filter by Source", options=["All"] + df["Source"].unique().tolist())
    if source_filter != "All":
        df = df[df["Source"] == source_filter]
    
    # Optional filtering by emotion
    emotion_filter = st.multiselect("Filter by Emotion", options=df["Emotion"].unique())
    if emotion_filter:
        df = df[df["Emotion"].isin(emotion_filter)]

    st.write("### Filtered Data", df)

    # Option to download filtered data
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=df.to_csv(index=False),
        file_name="filtered_creatives.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("Built for early testing, clustering, and scoring of creative inputs.")
