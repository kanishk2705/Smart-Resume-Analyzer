import streamlit as st

import api_client
from api_client import BackendError

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("🤖 Smart Resume Analyzer")

# --- 1. SESSION STATE ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- 2. SIDEBAR ---
gemini_configured = False
with st.sidebar:
    try:
        health = api_client.get_health()
        gemini_configured = health["gemini_configured"]
        st.success("Backend connected")
        st.info(f"AI Logic: SBERT ({health['embedding_model']}) + Gemini")
    except BackendError as e:
        st.error(str(e))

# --- 3. INPUTS ---
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
with col2:
    jd_input = st.text_area("Paste JD", height=200)

# --- 4. ANALYZE BUTTON ---
if st.button("Analyze Match"):
    if not uploaded_file or not jd_input.strip():
        st.warning("Upload a resume and paste a job description first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                st.session_state.analysis_results = api_client.analyze(
                    uploaded_file.name, uploaded_file.getvalue(), jd_input
                )
            except BackendError as e:
                st.session_state.analysis_results = None
                st.error(str(e))

# --- 5. DISPLAY RESULTS ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results

    st.divider()

    # Score Section
    score = results["match_score"]
    col_score, col_hero = st.columns([1, 2])

    with col_score:
        st.metric("Match Confidence", f"{score}%")
        if score >= 75:
            st.success("✅ High Match")
        elif score >= 50:
            st.warning("⚠️ Moderate Match")
        else:
            st.error("❌ Low Match")

    # Hero Sentences
    with col_hero:
        st.subheader("✨ Hero Sentences")
        if results["top_sentences"]:
            for sentence in results["top_sentences"]:
                st.info(f"**({int(sentence['score'] * 100)}%)** ...{sentence['text']}...")
        else:
            st.info("No strong sentences found.")

    # Missing Keywords
    st.divider()
    st.subheader("⚠️ Missing Keywords")
    if results["missing_keywords"]:
        st.write(", ".join([f"`{w}`" for w in results["missing_keywords"]]))

        # --- GEN AI SECTION ---
        st.divider()
        st.subheader("🎓 AI Personal Growth Plan")

        # API Key Logic (configured on the backend, or entered here)
        api_key = None
        if not gemini_configured:
            api_key = st.text_input("Enter Gemini API Key:", type="password")

        if st.button("Generate Learning Path"):
            if not gemini_configured and not api_key:
                st.error("Please enter an API Key.")
            else:
                status_msg = st.empty()
                status_msg.info("⏳ Consulting AI Coach... Please wait.")

                try:
                    recs = api_client.get_recommendations(results["missing_keywords"], api_key)
                except BackendError as e:
                    recs = None
                    st.error(str(e))
                finally:
                    status_msg.empty()

                if recs:
                    cols = st.columns(3)
                    for i, item in enumerate(recs):
                        with cols[i % 3]:
                            with st.container(border=True):
                                st.markdown(f"**📘 Learn {item['skill']}**")
                                st.caption(f"🔗 [{item['resource_name']}]({item['link']})")
                                st.success(f"🛠 **Project:** {item['project']}")
                elif recs is not None:
                    st.warning("No recommendations returned.")
    else:
        st.success("No missing keywords found.")
