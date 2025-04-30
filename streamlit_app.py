import streamlit as st
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.config import load_from_dict

# Page config
st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

# Steps setup
steps = ["Upload", "Scoring", "Keyword Planner", "Emotional Lens", "Explorer", "Export"]
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0

def go_next():
    if st.session_state.step_idx < len(steps) - 1:
        st.session_state.step_idx += 1

def go_back():
    if st.session_state.step_idx > 0:
        st.session_state.step_idx -= 1

current_step = steps[st.session_state.step_idx]

# UI Layout
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown(f"**Step {st.session_state.step_idx + 1} of {len(steps)}**")
st.sidebar.progress((st.session_state.step_idx + 1) / len(steps))
st.sidebar.radio("Jump to step", steps, index=st.session_state.step_idx, key="manual_step")

if st.session_state.manual_step != current_step:
    st.session_state.step_idx = steps.index(st.session_state.manual_step)
    current_step = st.session_state.manual_step

st.title(f"🧠 Step {st.session_state.step_idx + 1}: {current_step}")

# Step 1: Upload
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

# Step 2: Scoring
elif current_step == "Scoring":
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        df["score"] = df["creative_text"].apply(lambda x: len(str(x)) % 10 + 1)
        st.session_state.df = df
        st.success("✅ Creatives scored.")
        st.dataframe(df)
    else:
        st.warning("Upload data first.")

# Step 3: Keyword Planner
elif current_step == "Keyword Planner":
    config_dict = {
        "developer_token": st.secrets["google_ads"]["developer_token"],
        "client_id": st.secrets["google_ads"]["client_id"],
        "client_secret": st.secrets["google_ads"]["client_secret"],
        "refresh_token": st.secrets["google_ads"]["refresh_token"],
        "use_proto_plus": True
    }

    try:
        client = GoogleAdsClient.load_from_dict(config_dict)
    except Exception as e:
        st.error(f"Google Ads error: {e}")
        st.stop()

    customer_id = "1745036270"
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

# Step 4: Emotional Lens (placeholder logic)
elif current_step == "Emotional Lens":
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        df["emotion"] = df["creative_text"].apply(lambda x: "Curiosity" if "?" in str(x) else "Neutral")
        df["suggested_rewrite"] = df["creative_text"].apply(lambda x: f"🔥 {x}")
        st.session_state.df = df
        st.success("🧠 Emotion analysis added")
        st.dataframe(df)
    else:
        st.warning("Upload or generate creatives first.")

# Step 5: Explorer
elif current_step == "Explorer":
    if "df" in st.session_state:
        st.dataframe(st.session_state.df)
    else:
        st.warning("No data loaded.")

# Step 6: Export
elif current_step == "Export":
    if "df" in st.session_state:
        st.download_button("⬇️ Download Results", data=st.session_state.df.to_csv(index=False), file_name="creative_output.csv")
    else:
        st.warning("Nothing to export.")

# Navigation Buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.session_state.step_idx > 0:
        st.button("⬅️ Back", on_click=go_back)
with col3:
    if st.session_state.step_idx < len(steps) - 1:
        st.button("Next ➡️", on_click=go_next)
