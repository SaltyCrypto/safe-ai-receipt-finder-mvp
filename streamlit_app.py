import streamlit as st
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

@st.cache_resource
def load_ads_client():
    """
    Load the Google Ads client from Streamlit secrets.
    """
    return GoogleAdsClient.load_from_dict(st.secrets["google_ads"])

def test_connection():
    """
    Run a simple query to verify Google Ads API connectivity.
    """
    client = load_ads_client()
    ga_service = client.get_service("GoogleAdsService")
    try:
        query = "SELECT customer.id, customer.descriptive_name FROM customer LIMIT 1"
        response = ga_service.search_stream(
            customer_id=client.login_customer_id or st.secrets["google_ads"].get("login_customer_id"),
            query=query
        )
        for batch in response:
            for row in batch.results:
                st.success(f"✅ Connected! Account {row.customer.id} – {row.customer.descriptive_name}")
                return
        st.warning("⚠️ Connected but no data returned. Check your customer_id.")
    except GoogleAdsException as ex:
        st.error("❌ API call failed:")
        for error in ex.failure.errors:
            st.error(f"  • {error.message}")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

def list_customers():
    """
    List all accessible customers under the login customer ID.
    """
    client = load_ads_client()
    svc = client.get_service("CustomerService")
    response = svc.list_accessible_customers()
    st.write("**Accessible Customers:**")
    for resource in response.resource_names:
        st.write(f"- {resource}")

def main():
    st.title("OAuth Finalizer")
    st.markdown("Validate your Google Ads OAuth credentials and list accessible customers.")

    if st.button("Test Google Ads Connection"):
        test_connection()

    if st.button("List Accessible Customers"):
        list_customers()

if __name__ == "__main__":
    main()
