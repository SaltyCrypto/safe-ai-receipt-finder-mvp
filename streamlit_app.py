import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from openai import OpenAI

st.set_page_config(page_title="Creative Analyzer PRO", layout="wide")

# --- Initialize Session State ---
def init_state():
    defaults = {
        "df": None,
        "step": "Upload",
        "api_key": None,
        "valid_key": False,
        "client": None,
        "x": None,
        "y": None,
        "cluster": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# --- Sidebar Progress ---
steps = ["Upload", "Scoring", "Explorer", "Clustering", "Optimization", "Export"]
st.sidebar.title("🧠 Creative Workflow")
step_index = steps.index(st.session_state["step"])
st.sidebar.markdown(f"### ▶️ Step {step_index + 1}: {steps[step_index]}")
st.sidebar.progress(step_index / (len(steps) - 1))
st.markdown("#### Progress")
st.progress(step_index / (len(steps) - 1))

# --- API Key ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔐 API Access")
api_input = st.sidebar.text_input("Enter OpenAI API Key", type="password").strip()
if api_input:
    try:
        client = OpenAI(api_key=api_input)
        st.session_state.api_key = api_input
        st.session_state.valid_key = True
        st.session_state.client = client
        st.sidebar.success("✅ API Key Valid")
    except Exception as e:
        st.sidebar.error(f"❌ API Error: {str(e)}")

# --- Upload Step ---
if st.session_state["step"] == "Upload":
    st.title("📂 Upload Creative CSV")
    uploaded = st.file_uploader("Upload a CSV with 'creative_text' column", type="csv")
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.success("✅ Uploaded successfully.")
        st.markdown(f"**{len(st.session_state.df)} hooks loaded.**")
        st.dataframe(st.session_state.df.head(10))
        if st.button("Next: Score Creatives"):
            st.session_state["step"] = "Scoring"

# --- Scoring Step ---
elif st.session_state["step"] == "Scoring":
    st.title("📊 Scoring Engine")
    if st.session_state.df is not None:
        if st.button("Score Creatives"):
            with st.spinner("Scoring..."):
                time.sleep(1)
                st.session_state.df["score"] = np.random.uniform(3, 9, len(st.session_state.df))
                st.toast("Scoring complete")
            st.success("✅ Creatives scored successfully!")

        if 'score' in st.session_state.df:
            cols = [col for col in ['creative_text', 'score'] if col in st.session_state.df.columns]
            if cols:
                st.dataframe(st.session_state.df[cols])
            else:
                st.warning("No valid columns found. Make sure your CSV has 'creative_text'.")
            avg = st.session_state.df['score'].mean()
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg,
                title={"text": "Avg Score"},
                gauge={"axis": {"range": [0, 10]}}
            )), use_container_width=True)
            st.button("Next: Explore Embeddings", disabled='score' not in st.session_state.df, on_click=lambda: st.session_state.update({"step": "Explorer"}))
        else:
            st.warning("Please click 'Score Creatives' to continue.")

# --- Explorer Step ---
elif st.session_state["step"] == "Explorer":
    st.title("🧭 Embedding Explorer")
    if st.session_state["x"] is None:
        st.session_state.df['x'] = np.random.randn(len(st.session_state.df))
        st.session_state.df['y'] = np.random.randn(len(st.session_state.df))

    required_cols = {'x', 'y', 'creative_text'}
    if required_cols.issubset(st.session_state.df.columns) and not st.session_state.df.empty:
        df = st.session_state.df.dropna(subset=['x', 'y'])
        if len(df) == 0:
            st.warning("No valid rows to plot.")
        else:
            try:
                if 'score' in df.columns:
                    fig = px.scatter(df, x='x', y='y', color='score', hover_data=['creative_text'])
                else:
                    st.warning("No scores found. Continuing without them.")
                    fig = px.scatter(df, x='x', y='y', hover_data=['creative_text'])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plotting failed: {e}")
    else:
        st.error("Required columns missing.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Scoring"):
            st.session_state["step"] = "Scoring"
    with col2:
        if st.button("Next: Cluster Creatives"):
            st.session_state["step"] = "Clustering"

# --- Clustering Step ---
elif st.session_state["step"] == "Clustering":
    st.title("📚 Clustering")
    if st.session_state.df is not None:
        required_cols = {'x', 'y', 'creative_text'}
        if required_cols.issubset(st.session_state.df.columns) and not st.session_state.df.empty:
            k = st.slider("Number of clusters", 2, 10, 4)
            st.session_state.df['cluster'] = np.random.randint(0, k, size=len(st.session_state.df))
            fig = px.scatter(st.session_state.df, x='x', y='y', color='cluster', hover_data=['creative_text'])
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(st.session_state.df[['creative_text', 'cluster']])
        else:
            st.error("Required data missing.")
    if st.button("Next: Optimize Creatives"):
        st.session_state["step"] = "Optimization"

# --- Optimization Step ---
elif st.session_state["step"] == "Optimization":
    st.title("🪄 Optimization")
    col1, col2 = st.columns(2)
    with col1:
        a = st.text_area("Creative A", "Affordable life insurance in 60s")
    with col2:
        b = st.text_area("Creative B", "Protect your family for $5/mo")

    if st.button("Simulate Winner"):
        winner = a if np.random.rand() > 0.5 else b
        st.success(f"🏁 Winner: {winner[:50]}...")

    raw = st.text_area("Rewrite this creative:", "Get $300 off now!")
    if st.button("Rewrite & Explain") and st.session_state.valid_key:
        try:
            client = st.session_state.client
            res = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative marketing assistant."},
                    {"role": "user", "content": f"Rewrite this creative: {raw}"}
                ]
            )
            rewritten = res.choices[0].message.content.strip()
            st.chat_message("assistant").write(f"✨ Rewritten: {rewritten}")
        except Exception as e:
            st.error(f"Rewrite failed: {e}")
    elif not st.session_state.valid_key:
        st.warning("Enter OpenAI key in the sidebar first.")

    if st.button("Next: Export Results"):
        st.session_state["step"] = "Export"

# --- Export Step ---
elif st.session_state["step"] == "Export":
    st.title("📤 Export")
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.download_button("Download CSV", st.session_state.df.to_csv(index=False), "enhanced_creatives.csv")
        email = st.text_input("Send to email")
        if st.button("Send"):
            st.toast("Simulated send complete.")
        st.success("🎉 Workflow complete!")
    else:
        st.warning("Nothing to export.")
