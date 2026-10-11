import streamlit as st

import api_client
from api_client import BackendError

STATUS_ICONS = {
    "matched": "✅", "related": "🟡", "missing": "❌",
    "strong": "🟢", "partial": "🟡", "weak": "🔴",
}

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
    col_score, col_breakdown = st.columns([1, 2])

    with col_score:
        st.metric("Overall Match", f"{score:.0f}%")
        if score >= 75:
            st.success("✅ High Match")
        elif score >= 50:
            st.warning("⚠️ Moderate Match")
        else:
            st.error("❌ Low Match")
        if results["sections_found"]:
            st.caption("Sections detected: " + ", ".join(results["sections_found"]))
        else:
            st.caption("No section headings detected; the resume was scored as one block.")

    # Score Breakdown
    with col_breakdown:
        st.subheader("📊 Score Breakdown")
        for component in results["components"]:
            share = f"{component['weight']:.0%} of score"
            if component["score"] is None:
                st.markdown(f"**{component['label']}** — not assessed ({share})")
            else:
                st.markdown(f"**{component['label']}** — {component['score']:.0f}% ({share})")
                st.progress(component["score"] / 100)
            st.caption(component["detail"])

    # Skills
    st.divider()
    st.subheader("🧩 Skills")
    skills = results["skills"]
    missing_skills = [s["name"] for s in skills if s["status"] == "missing"]
    if skills:
        categories = list(dict.fromkeys(s["category"] for s in skills))
        for category in categories:
            chips = []
            for skill in (s for s in skills if s["category"] == category):
                chip = f"{STATUS_ICONS[skill['status']]} `{skill['name']}`"
                if skill["related_to"]:
                    chip += f" (you have {skill['related_to']})"
                chips.append(chip)
            st.markdown(f"**{category}:** " + " · ".join(chips))
        st.caption("✅ found in resume · 🟡 a related skill was found · ❌ missing")
    else:
        st.info("No known skills were found in the job description.")
    if results["additional_skills"]:
        with st.expander("Other skills on your resume that this job did not ask for"):
            st.write(", ".join(f"`{name}`" for name in results["additional_skills"]))

    # Requirement Coverage
    st.divider()
    st.subheader("🔍 Requirement Coverage")
    st.caption("Each job requirement, with the resume sentence that supports it best.")
    st.dataframe(
        [
            {
                "Match": STATUS_ICONS[r["status"]] + " " + r["status"].title(),
                "Job requirement": r["requirement"],
                "Best evidence in resume": r["evidence"],
                "Similarity": round(r["similarity"], 2),
            }
            for r in results["requirements"]
        ],
        hide_index=True,
        width="stretch",
    )

    # Hero Sentences
    st.subheader("✨ Hero Sentences")
    if results["top_sentences"]:
        for sentence in results["top_sentences"]:
            st.info(f"**({int(sentence['score'] * 100)}%)** {sentence['text']}")
    else:
        st.info("No strong sentences found.")

    # --- GEN AI SECTION ---
    st.divider()
    st.subheader("🎓 AI Personal Growth Plan")
    if missing_skills:
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
                    recs = api_client.get_recommendations(missing_skills, api_key)
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
        st.success("No missing skills to plan for.")
