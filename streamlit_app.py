elif current_step == "Keyword Planner":
    st.subheader("🔍 Google Ads Keyword Planner")

    # Config setup from secrets
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
        st.error(f"❌ Google Ads client load failed: {e}")
        st.stop()

    customer_id = "2933192176"

    st.markdown("Enter one keyword phrase per line:")
    keyword_input = st.text_area("💡 Seed Keywords", "life insurance\nterm life insurance\nbest policy for parents")

    geo = st.selectbox("🌍 Geo Target", {
        "United States": "geoTargetConstants/2840",
        "UK": "geoTargetConstants/2826",
        "Canada": "geoTargetConstants/2124"
    }.items(), format_func=lambda x: x[0])

    lang = st.selectbox("🈯 Language", {"English": "1000", "Spanish": "1003"}.items(), format_func=lambda x: x[0])

    if st.button("🚀 Fetch Keyword Ideas"):
        try:
            keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
            request = client.get_type("GenerateKeywordIdeasRequest")

            request.customer_id = customer_id
            request.language = lang[1]
            request.geo_target_constants.append(geo[1])

            # Parse multi-line input into list of phrases
            keywords = [kw.strip() for kw in keyword_input.split("\n") if kw.strip()]
            if not keywords:
                st.warning("⚠️ Please enter at least one keyword phrase.")
                st.stop()

            # Initialize keyword_seed and attach
            keyword_seed = client.get_type("KeywordSeed")
            keyword_seed.keywords.extend(keywords)
            request.keyword_seed = keyword_seed

            # Make the call
            response = keyword_plan_idea_service.generate_keyword_ideas(request=request)

            result = []
            for idea in response:
                metrics = idea.keyword_idea_metrics
                result.append({
                    "Keyword": idea.text,
                    "Searches/mo": metrics.avg_monthly_searches,
                    "Competition": metrics.competition.name,
                    "Low CPC ($)": round(metrics.low_top_of_page_bid_micros / 1_000_000, 2),
                    "High CPC ($)": round(metrics.high_top_of_page_bid_micros / 1_000_000, 2)
                })

            df = pd.DataFrame(result)
            st.session_state.df = df
            st.success("✅ Keyword ideas retrieved")
            st.dataframe(df)

        except Exception as e:
            st.error(f"❌ Keyword Planner error: {e}")