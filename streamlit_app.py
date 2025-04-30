import streamlit as st
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.config import load_from_dict

st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

steps = ["Upload", "Scoring", "Keyword Planner", "Emotional Lens", "Explorer", "Export", "GPT Rewrite", "Emotion Scoring", "Clustering"]
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
        "login_customer_id": "9816127168",  # Manager account ID
        "use_proto_plus": True
    }

    try:
        client = GoogleAdsClient.load_from_dict(config_dict)
    except Exception as e:
        st.error(f"Google Ads error: {e}")
        st.stop()

    customer_id = "2933192176"  # Newly created account

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
            request.keyword_seed.keywords.append(keyword)

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

# Placeholder for remaining steps
elif current_step == "Emotional Lens":
    st.write("Emotional Lens Tab Placeholder")
elif current_step == "Explorer":
    if "df" in st.session_state:
        st.dataframe(st.session_state.df)
elif current_step == "Export":
    if "df" in st.session_state:
        st.download_button("⬇️ Download Results", data=st.session_state.df.to_csv(index=False), file_name="creative_output.csv")
elif current_step == "GPT Rewrite":
    st.write("GPT Rewrite Placeholder")
elif current_step == "Emotion Scoring":
    st.write("Emotion Scoring Placeholder")
elif current_step == "Clustering":
    st.write("Clustering Tab Placeholder")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.session_state.step_idx > 0:
        st.button("⬅️ Back", on_click=go_back)
with col3:
    if st.session_state.step_idx < len(steps) - 1:
        st.button("Next ➡️", on_click=go_next)
