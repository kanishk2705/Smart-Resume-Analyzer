import csv
import io

import streamlit as st

import api_client
from api_client import BackendError

STATUS_ICONS = {
    "matched": "✅", "related": "🟡", "missing": "❌",
    "strong": "🟢", "partial": "🟡", "weak": "🔴",
    "pass": "✅", "warn": "⚠️", "fail": "❌",
}

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("🤖 Smart Resume Analyzer")

# --- 1. SESSION STATE ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "ranking_results" not in st.session_state:
    st.session_state.ranking_results = None
if "reports" not in st.session_state:
    st.session_state.reports = {}  # analysis id -> PDF bytes

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


def match_band(score: float) -> None:
    if score >= 75:
        st.success("✅ High Match")
    elif score >= 50:
        st.warning("⚠️ Moderate Match")
    else:
        st.error("❌ Low Match")


def show_score(results: dict) -> None:
    col_score, col_breakdown = st.columns([1, 2])

    with col_score:
        st.metric("Overall Match", f"{results['match_score']:.0f}%")
        match_band(results["match_score"])
        st.metric("ATS Format Score", f"{results['ats']['score']:.0f}%")
        if results["sections_found"]:
            st.caption("Sections detected: " + ", ".join(results["sections_found"]))
        else:
            st.caption("No section headings detected; the resume was scored as one block.")

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


def show_skills(results: dict) -> None:
    st.subheader("🧩 Skills")
    skills = results["skills"]
    if skills:
        for category in dict.fromkeys(s["category"] for s in skills):
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


def show_requirements(results: dict) -> None:
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

    st.subheader("✨ Hero Sentences")
    if results["top_sentences"]:
        for sentence in results["top_sentences"]:
            st.info(f"**({int(sentence['score'] * 100)}%)** {sentence['text']}")
    else:
        st.info("No strong sentences found.")


def show_ats(results: dict) -> None:
    st.subheader("📋 ATS Format Check")
    st.caption("Whether an applicant tracking system can read this resume. Independent of the job.")
    for check in results["ats"]["checks"]:
        st.markdown(f"{STATUS_ICONS[check['status']]} **{check['label']}** — {check['message']}")
        if check["examples"]:
            with st.expander("Examples from your resume"):
                for example in check["examples"]:
                    st.markdown(f"- {example}")


def show_learning_path(results: dict, key: str) -> None:
    st.subheader("🎓 AI Personal Growth Plan")
    missing_skills = [s["name"] for s in results["skills"] if s["status"] == "missing"]
    if not missing_skills:
        st.success("No missing skills to plan for.")
        return

    # API Key Logic (configured on the backend, or entered here)
    api_key = None
    if not gemini_configured:
        api_key = st.text_input("Enter Gemini API Key:", type="password", key=f"{key}_api_key")

    if st.button("Generate Learning Path", key=f"{key}_learning_path"):
        if not gemini_configured and not api_key:
            st.error("Please enter an API Key.")
            return

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


def show_report_download(results: dict, key: str) -> None:
    analysis_id = results["id"]
    if analysis_id not in st.session_state.reports:
        try:
            st.session_state.reports[analysis_id] = api_client.get_report(analysis_id)
        except BackendError as e:
            st.caption(f"PDF report unavailable: {e}")
            return
    st.download_button(
        "📄 Download PDF report",
        st.session_state.reports[analysis_id],
        file_name=f"resume-report-{analysis_id}.pdf",
        mime="application/pdf",
        key=f"{key}_report",
    )


def show_analysis(results: dict, key: str) -> None:
    """Renders one analysis. `key` keeps widgets apart when two analyses are on screen."""
    st.divider()
    st.caption(f"{results['resume_filename']} · {results['job_title']}")
    show_score(results)
    show_report_download(results, key)
    st.divider()
    show_skills(results)
    st.divider()
    show_requirements(results)
    st.divider()
    show_ats(results)
    st.divider()
    show_learning_path(results, key)


def ranking_csv(candidates: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    labels = [c["label"] for c in candidates[0]["components"]]
    writer.writerow(["Rank", "Resume", "Overall", *labels, "Matched", "Related", "Missing"])
    for candidate in candidates:
        scores = ["" if c["score"] is None else c["score"] for c in candidate["components"]]
        skills = [
            "; ".join(candidate[f"{status}_skills"])
            for status in ("matched", "related", "missing")
        ]
        writer.writerow(
            [candidate["rank"], candidate["filename"], candidate["match_score"], *scores, *skills]
        )
    return buffer.getvalue()


def show_ranking(ranking: dict) -> None:
    st.divider()
    st.caption(ranking["job_title"])
    for failure in ranking["failed"]:
        st.warning(f"**{failure['filename']}** was skipped: {failure['error']}")

    candidates = ranking["candidates"]
    if not candidates:
        st.error("None of the resumes could be read.")
        return

    st.subheader("🏆 Ranking")
    st.dataframe(
        [
            {
                "Rank": candidate["rank"],
                "Resume": candidate["filename"],
                "Overall": candidate["match_score"],
                **{c["label"]: c["score"] for c in candidate["components"]},
                "Missing skills": ", ".join(candidate["missing_skills"]) or "—",
            }
            for candidate in candidates
        ],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "⬇️ Download ranking as CSV", ranking_csv(candidates), "ranking.csv", "text/csv"
    )

    for candidate in candidates:
        title = f"#{candidate['rank']} {candidate['filename']} — {candidate['match_score']:.0f}%"
        with st.expander(title):
            for component in candidate["components"]:
                st.markdown(f"**{component['label']}:** {component['detail']}")
            for status in ("matched", "related", "missing"):
                names = candidate[f"{status}_skills"]
                if names:
                    st.markdown(
                        f"{STATUS_ICONS[status]} **{status.title()}:** "
                        + ", ".join(f"`{name}`" for name in names)
                    )


def show_history() -> None:
    try:
        analyses = api_client.get_history()
    except BackendError as e:
        st.error(str(e))
        return
    if not analyses:
        st.info("No analyses yet. Results from the first tab are saved here automatically.")
        return

    # Compare each analysis with the previous one for the same job, to show progress
    previous_score: dict[str, float] = {}
    rows = []
    for analysis in reversed(analyses):  # oldest first
        earlier = previous_score.get(analysis["job_title"])
        rows.append({
            "ID": analysis["id"],
            "When (UTC)": analysis["created_at"][:16].replace("T", " "),
            "Resume": analysis["resume_filename"],
            "Job": analysis["job_title"],
            "Match": analysis["match_score"],
            "Change": None if earlier is None else round(analysis["match_score"] - earlier, 2),
            "ATS": analysis["ats_score"],
        })
        previous_score[analysis["job_title"]] = analysis["match_score"]
    st.dataframe(rows[::-1], hide_index=True, width="stretch")
    st.caption("Change is the difference from the previous analysis for the same job.")

    labels = {
        a["id"]: f"#{a['id']} · {a['resume_filename']} · {a['job_title']}" for a in analyses
    }
    selected = st.selectbox("Open a past analysis", list(labels), format_func=labels.get)
    col_open, col_delete = st.columns(2)
    if col_open.button("Open"):
        try:
            st.session_state.history_view = api_client.get_analysis(selected)
        except BackendError as e:
            st.error(str(e))
    if col_delete.button("Delete"):
        try:
            api_client.delete_analysis(selected)
            st.session_state.history_view = None
            st.rerun()
        except BackendError as e:
            st.error(str(e))

    if st.session_state.get("history_view"):
        show_analysis(st.session_state.history_view, key="history")


# --- 3. TABS ---
tab_analyze, tab_rank, tab_history = st.tabs(
    ["Analyze my resume", "Rank candidates", "History"]
)

with tab_analyze:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
    with col2:
        jd_input = st.text_area("Paste JD", height=200)

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

    if st.session_state.analysis_results:
        show_analysis(st.session_state.analysis_results, key="analyze")

with tab_rank:
    st.caption("Upload several resumes to rank them against one job description.")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_resumes = st.file_uploader(
            "Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True
        )
    with col2:
        rank_jd_input = st.text_area("Paste JD", height=200, key="rank_jd")

    if st.button("Rank Candidates"):
        if not uploaded_resumes or not rank_jd_input.strip():
            st.warning("Upload at least one resume and paste a job description first.")
        else:
            with st.spinner(f"Ranking {len(uploaded_resumes)} resumes..."):
                try:
                    st.session_state.ranking_results = api_client.rank(
                        [(f.name, f.getvalue()) for f in uploaded_resumes], rank_jd_input
                    )
                except BackendError as e:
                    st.session_state.ranking_results = None
                    st.error(str(e))

    if st.session_state.ranking_results:
        show_ranking(st.session_state.ranking_results)

with tab_history:
    show_history()
