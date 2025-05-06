import streamlit as st
import pandas as pd

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.oauth2 import AuthCodeGoogleAdsCredential
from google.ads.googleads.v14.enums.types import KeywordPlanNetworkEnum

st.set_page_config(page_title="🔒 Google Ads OAuth Finalizer + Diagnostic", layout="wide")
st.title("🔒 Google Ads OAuth Finalizer + Diagnostic")

#
# 1. CONFIG VALIDATION & OAUTH FLOW
#
REQUIRED = ["developer_token", "client_id", "client_secret"]
for key in REQUIRED:
    if key not in st.secrets or not st.secrets[key]:
        st.error(f"⚠️ Missing secret: `{key}` in Streamlit secrets.toml")
        st.stop()

developer_token = st.secrets["developer_token"]
client_id       = st.secrets["client_id"]
client_secret   = st.secrets["client_secret"]
refresh_token   = st.secrets.get("refresh_token", "")

if not refresh_token:
    st.markdown("### 1. Authorize your application")
    cred = AuthCodeGoogleAdsCredential(
        client_id=client_id,
        client_secret=client_secret,
        developer_token=developer_token,
    )
    auth_url = cred.get_authorization_url(
        scopes=["https://www.googleapis.com/auth/adwords"],
    )
    st.markdown(f"[1. Click here to authorize →]({auth_url})")
    code = st.text_input("2. Paste the authorization code here")
    if code:
        creds = cred.get_credentials_from_code(code=code)
        st.success("✅ Refresh token generated!")
        st.code(f"refresh_token: \"{creds.refresh_token}\"")
        st.info("Copy that value into your streamlit `secrets.toml` under `refresh_token` then re‑run.")
        st.stop()

# build client
config_dict = {
    "developer_token": developer_token,
    "client_id":       client_id,
    "client_secret":   client_secret,
    "refresh_token":   refresh_token,
}
client = GoogleAdsClient.load_from_dict(config_dict, version="v14")


#
# 2. SERVICE SELECTOR
#
st.markdown("## 2. Select Service and Run Diagnostic")
service = st.selectbox(
    "Choose Google Ads service:",
    ["CustomerService", "GoogleAdsService", "KeywordPlanIdeaService"],
)

#
# 3a. CustomerService
#
if service == "CustomerService":
    st.markdown("### List Accessible Customers")
    if st.button("Run CustomerService"):
        try:
            svc = client.get_service("CustomerService")
            resp = svc.list_accessible_customers()
            ids = resp.resource_names
            df = pd.DataFrame({"resource_name": ids})
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Error: {e}")

#
# 3b. GoogleAdsService
#
elif service == "GoogleAdsService":
    st.markdown("### Run a GAQL Query")
    customer_id = st.text_input("Customer ID", value="", placeholder="e.g. 1234567890")
    gaql = st.text_area("GAQL Query", value="SELECT campaign.id, campaign.name FROM campaign LIMIT 5")
    if st.button("Run GAQL Query"):
        if not customer_id.strip():
            st.error("Please enter a Customer ID.")
        else:
            try:
                ga_svc = client.get_service("GoogleAdsService")
                stream = ga_svc.search_stream(customer_id=customer_id, query=gaql)
                rows = []
                for batch in stream:
                    for row in batch.results:
                        d = {f.name: getattr(row, f.name) for f in row._pb.DESCRIPTOR.fields}
                        rows.append(d)
                st.dataframe(pd.DataFrame(rows))
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

#
# 3c. KeywordPlanIdeaService
#
elif service == "KeywordPlanIdeaService":
    st.markdown("### Keyword Planner: Generate Keyword Ideas")
    customer_id  = st.text_input("Customer ID", value="", placeholder="e.g. 1234567890")
    seed_kw      = st.text_area("Seed Keywords (comma‑separated)", "running shoes, trail running")
    location_ids = st.text_input("Location Criterion IDs", "2840")  # 2840=US
    language_id  = st.text_input("Language Criterion ID", "1000")   # 1000=English
    network_opt  = st.selectbox("Network", ["GOOGLE_SEARCH", "GOOGLE_SEARCH_AND_PARTNERS"])

    if st.button("Generate Keyword Ideas"):
        if not customer_id.strip():
            st.error("Please enter a Customer ID.")
        else:
            try:
                kp_svc = client.get_service("KeywordPlanIdeaService")
                request = {
                    "customer_id": customer_id,
                    "language": int(language_id),
                    "geo_target_constants": [f"geoTargetConstants/{int(x.strip())}"
                                             for x in location_ids.split(",")],
                    "keyword_seed": {"keywords": [k.strip() for k in seed_kw.split(",")]},
                    "network": getattr(KeywordPlanNetworkEnum, network_opt),
                }
                response = kp_svc.generate_keyword_ideas(request=request)

                rows = []
                for idea in response:
                    m = idea.keyword_idea_metrics
                    rows.append({
                        "Keyword": idea.text,
                        "Avg. Monthly Searches": m.avg_monthly_searches,
                        "Competition": m.competition.name,
                        "Top of page bid (low)": m.low_top_of_page_bid_micros / 1e6,
                        "Top of page bid (high)": m.high_top_of_page_bid_micros / 1e6,
                    })

                st.dataframe(pd.DataFrame(rows))

            except Exception as e:
                st.error(f"❌ Keyword Planner failed: {e}")
