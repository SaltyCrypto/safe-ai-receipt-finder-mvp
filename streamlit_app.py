import streamlit as st
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.config import load_from_dict

st.set_page_config(page_title="Creative Intelligence - Google Ads", layout="wide")

if "step" not in st.session_state:
    st.session_state["step"] = "Upload"

st.sidebar.title("🧭 Navigation")
st.session_state["step"] = st.sidebar.radio("Go to step", ["Upload", "Scoring", "Keyword Planner", "Explorer"])

# Step 1: Upload creatives
if st.session_state["step"] == "Upload":
    st.title("📤 Upload Creative CSV")
    uploaded_file = st.file_uploader("Upload a CSV with a 'creative_text' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Text" in df.columns:
            df.rename(columns={"Text": "creative_text"}, inplace=True)
        if "creative_text" not in df.columns:
            st.error("CSV must contain a 'creative_text' column.")
        else:
            st.session_state.df = df
            st.success("✅ Creatives uploaded successfully!")
            st.dataframe(df)

# Step 2: Scoring (placeholder)
elif st.session_state["step"] == "Scoring":
    st.title("📊 Scoring Engine")
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        df["score"] = df["creative_text"].apply(lambda x: len(str(x)) % 10 + 1)  # Placeholder scoring logic
        st.session_state.df = df
        st.dataframe(df)
    else:
        st.warning("⚠️ Please upload creative data first.")

# Step 3: Keyword Planner
elif st.session_state["step"] == "Keyword Planner":
    st.title("🔍 Keyword Planner Explorer")

    config_dict = {
        "developer_token": st.secrets["google_ads"]["developer_token"],
        "client_id": st.secrets["google_ads"]["client_id"],
        "client_secret": st.secrets["google_ads"]["client_secret"],
        "refresh_token": st.secrets["google_ads"]["refresh_token"],
        "use_proto_plus": True
    }

    try:
        client_gads = GoogleAdsClient.load_from_dict(config_dict)
    except Exception as e:
        st.error(f"Google Ads connection failed: {e}")
        st.stop()

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
                if (
                    metrics.low_top_of_page_bid_micros < min_cpc
                    or metrics.high_top_of_page_bid_micros > max_cpc
                ):
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

        except Exception as e:
            st.error(f"Google Ads API Error: {e}")

# Step 4: Explorer (placeholder)
elif st.session_state["step"] == "Explorer":
    st.title("🔎 Creative Explorer")
    if "df" in st.session_state:
        st.dataframe(st.session_state.df)
    else:
        st.warning("⚠️ No data loaded.")
