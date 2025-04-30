import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from openai import OpenAI
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# --- Page setup ---
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

# --- Step Navigation ---
steps = ["Upload", "Scoring", "Keyword Planner", "Explorer", "Clustering", "Optimization", "Export"]
step_index = steps.index(st.session_state["step"])
st.sidebar.title("🧠 Creative Workflow")
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

# --- Step 1: Upload ---
if st.session_state["step"] == "Upload":
    st.title("📂 Upload Creative CSV")
    uploaded = st.file_uploader("Upload a CSV with 'creative_text' column", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "Text" in df.columns and "creative_text" not in df.columns:
            df.rename(columns={"Text": "creative_text"}, inplace=True)
            st.info("🛠️ Renamed 'Text' column to 'creative_text' automatically.")
        elif "creative_text" not in df.columns:
            st.warning("❗ No 'creative_text' column found.")
        st.session_state.df = df
        st.success("✅ Uploaded successfully.")
        st.markdown(f"**{len(df)} hooks loaded.**")
        st.dataframe(df.head(10))
        if st.button("Next: Score Creatives"):
            st.session_state["step"] = "Scoring"

# --- Step 2: Scoring ---
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
            avg = st.session_state.df['score'].mean()
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg,
                title={"text": "Avg Score"},
                gauge={"axis": {"range": [0, 10]}}
            )), use_container_width=True)
            if st.button("Next: Keyword Planner"):
                st.session_state["step"] = "Keyword Planner"
        else:
            st.warning("Please click 'Score Creatives' to continue.")
    if st.button("⬅️ Back to Upload"):
        st.session_state["step"] = "Upload"

# --- Step 3: Keyword Planner ---
elif st.session_state["step"] == "Keyword Planner":
    st.title("🔍 Keyword Planner Explorer")
    client_gads = GoogleAdsClient.load_from_storage("google-ads.yaml")
    customer_id = "1745036270"

    seed_keyword = st.text_input("💡 Seed Keyword", "life insurance")
    geo_map = {
        "United States": "geoTargetConstants/2840",
        "United Kingdom": "geoTargetConstants/2826",
        "Canada": "geoTargetConstants/2124",
        "Australia": "geoTargetConstants/2036",
    }
    geo = st.selectbox("🌍 GEO", list(geo_map.keys()))
    geo_target = geo_map[geo]
    lang_map = {"English": "1000", "Spanish": "1003", "French": "1002"}
    lang = st.selectbox("🈯 Language", list(lang_map.keys()))
    lang_code = lang_map[lang]
    min_comp = st.selectbox("🎯 Min Competition", ["LOW", "MEDIUM", "HIGH", "ANY"], index=3)
    min_cpc = st.number_input("💰 Min CPC (μ)", 0, value=0)
    max_cpc = st.number_input("💰 Max CPC (μ)", 0, value=10_000_000)

    if st.button("🔎 Fetch Keyword Ideas"):
        try:
            kp_service = client_gads.get_service("KeywordPlanIdeaService")
            request = client_gads.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = customer_id
            request.language = lang_code
            request.geo_target_constants.append(geo_target)
            request.keyword_seed.keywords.append(seed_keyword)

            response = kp_service.generate_keyword_ideas(request=request)
            results = []
            for idea in response:
                metrics = idea.keyword_idea_metrics
                if min_comp != "ANY" and metrics.competition.name != min_comp:
                    continue
                if metrics.low_top_of_page_bid_micros < min_cpc or metrics.high_top_of_page_bid_micros > max_cpc:
                    continue
                results.append({
                    "Keyword": idea.text,
                    "Avg Monthly Searches": metrics.avg_monthly_searches,
                    "Competition": metrics.competition.name,
                    "Low CPC (μ)": metrics.low_top_of_page_bid_micros,
                    "High CPC (μ)": metrics.high_top_of_page_bid_micros
                })
            df = pd.DataFrame(results)
            st.session_state.df = df
            st.dataframe(df)
            st.success(f"✅ Pulled {len(df)} keyword ideas")
        except GoogleAdsException as e:
            st.error(f"Google Ads API Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    if st.button("⬅️ Back to Scoring"):
        st.session_state["step"] = "Scoring"
    if st.button("Next: Embedding Explorer"):
        st.session_state["step"] = "Explorer"

# --- Step 4: Explorer ---
elif st.session_state["step"] == "Explorer":
    st.title("🧭 Embedding Explorer")
    if st.session_state["x"] is None and st.session_state.df is not None:
        st.session_state.df['x'] = np.random.randn(len(st.session_state.df))
        st.session_state.df['y'] = np.random.randn(len(st.session_state.df))
    if {'x', 'y', 'creative_text'}.issubset(st.session_state.df.columns):
        df = st.session_state.df.dropna(subset=['x', 'y'])
        if len(df) == 0:
            st.warning("No valid rows to plot.")
        else:
            fig = px.scatter(df, x='x', y='y', color=df.get('score'), hover_data=['creative_text'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required data missing.")
    if st.button("⬅️ Back to Keyword Planner"):
        st.session_state["step"] = "Keyword Planner"
    if st.button("Next: Cluster Creatives"):
        st.session_state["step"] = "Clustering"

# --- Step 5: Clustering ---
elif st.session_state["step"] == "Clustering":
    st.title("📚 Clustering")
    if st.session_state.df is not None and {'x', 'y', 'creative_text'}.issubset(st.session_state.df.columns):
        k = st.slider("Number of clusters", 2, 10, 4)
        st.session_state.df['cluster'] = np.random.randint(0, k, size=len(st.session_state.df))
        fig = px.scatter(st.session_state.df, x='x', y='y', color='cluster', hover_data=['creative_text'])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(st.session_state.df[['creative_text', 'cluster']])
    if st.button("⬅️ Back to Explorer"):
        st.session_state["step"] = "Explorer"
    if st.button("Next: Optimize Creatives"):
        st.session_state["step"] = "Optimization"

# --- Step 6: Optimization ---
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
            res = st.session_state.client.chat.completions.create(
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
    if st.button("⬅️ Back to Clustering"):
        st.session_state["step"] = "Clustering"
    if st.button("Next: Export Results"):
        st.session_state["step"] = "Export"

# --- Step 7: Export ---
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
    if st.button("⬅️ Back to Optimization"):
        st.session_state["step"] = "Optimization"
