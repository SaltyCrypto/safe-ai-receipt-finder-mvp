import streamlit as st
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.config import load_from_dict

st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] .css-1d391kg {
        font-size: 1.1rem;
        padding-left: 0.5rem;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 1.3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

steps = ["Upload", "Scoring", "Keyword Planner", "Explorer", "Emotional Lens + Scoring", "GPT Rewrite", "Clustering", "Export"]

if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0

def go_next():
    if st.session_state.step_idx < len(steps) - 1:
        st.session_state.step_idx += 1

def go_back():
    if st.session_state.step_idx > 0:
        st.session_state.step_idx -= 1

current_step = steps[st.session_state.step_idx]
st.sidebar.title("🧭 Navigation")
st.sidebar.progress((st.session_state.step_idx + 1) / len(steps))
st.sidebar.radio("Jump to step", steps, index=st.session_state.step_idx, key="manual_step")

if st.session_state.manual_step != current_step:
    st.session_state.step_idx = steps.index(st.session_state.manual_step)
    current_step = st.session_state.manual_step

st.title(f"🧠 Step {st.session_state.step_idx + 1}: {current_step}")

if current_step == "Upload":
    file = st.file_uploader("📤 Upload CSV with 'creative_text' or 'Text' column", type="csv")
    if file:
        df = pd.read_csv(file)
        if "Text" in df.columns:
            df.rename(columns={"Text": "creative_text"}, inplace=True)
        if "creative_text" not in df.columns:
            st.error("Missing 'creative_text' column.")
        else:
            st.session_state.df = df
            st.success("✅ File uploaded!")
            st.dataframe(df)

elif current_step == "Scoring":
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        df["score"] = df["creative_text"].apply(lambda x: len(str(x)) % 10 + 1)
        st.session_state.df = df
        st.success("✅ Creatives scored.")
        st.dataframe(df)
    else:
        st.warning("Upload data first.")

elif current_step == "Keyword Planner":
    config_dict = {
        "developer_token": st.secrets["google_ads"]["developer_token"],
        "client_id": st.secrets["google_ads"]["client_id"],
        "client_secret": st.secrets["google_ads"]["client_secret"],
        "refresh_token": st.secrets["google_ads"]["refresh_token"],
        "login_customer_id": "9816127168",
        "use_proto_plus": True
    }

    try:
        client = GoogleAdsClient.load_from_dict(config_dict)
    except Exception as e:
        st.error(f"Google Ads error: {e}")
        st.stop()

    customer_id = "2933192176"

    keyword = st.text_input("💡 Seed keyword", "life insurance")
    geo = st.selectbox("🌍 Geo", {
        "United States": "geoTargetConstants/2840",
        "UK": "geoTargetConstants/2826",
        "Canada": "geoTargetConstants/2124"
    }.items())
    lang = st.selectbox("🈯 Language", {"English": "1000", "Spanish": "1003"}.items())

    if st.button("🔎 Fetch Keyword Ideas"):
        try:
            service = client.get_service("KeywordPlanIdeaService")
            request = client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = customer_id
            request.language = lang[1]
            request.geo_target_constants.append(geo[1])
            keyword_seed = client.get_type("KeywordSeed")
            keyword_seed.keywords.append(keyword)
            request.keyword_seed = keyword_seed

            ideas = service.generate_keyword_ideas(request=request)
            result = [{
                "Keyword": idea.text,
                "Searches": idea.keyword_idea_metrics.avg_monthly_searches,
                "Competition": idea.keyword_idea_metrics.competition.name,
                "Low CPC": idea.keyword_idea_metrics.low_top_of_page_bid_micros,
                "High CPC": idea.keyword_idea_metrics.high_top_of_page_bid_micros
            } for idea in ideas]
            df = pd.DataFrame(result)
            st.session_state.df = df
            st.success("✅ Keywords pulled")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Keyword error: {e}")

# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.session_state.step_idx > 0:
        st.button("⬅️ Back", on_click=go_back)
with col3:
    if st.session_state.step_idx < len(steps) - 1:
        st.button("Next ➡️", on_click=go_next)
