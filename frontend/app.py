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

# ATS checks whose examples are statements the AI coach can reword
REWRITABLE_CHECKS = ("quantified", "verbs", "bullet_length", "pronouns")

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("🤖 Smart Resume Analyzer")

# --- 1. SESSION STATE ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
    # A link ending in ?analysis=<id> opens that saved analysis
    linked_id = st.query_params.get("analysis", "")
    if linked_id.isdigit():
        try:
            st.session_state.analysis_results = api_client.get_analysis(int(linked_id))
        except BackendError:
            pass  # A stale link just opens the app as normal
if "ranking_results" not in st.session_state:
    st.session_state.ranking_results = None
if "reports" not in st.session_state:
    st.session_state.reports = {}  # analysis id -> PDF bytes
if "coaching" not in st.session_state:
    st.session_state.coaching = {}  # (analysis id, kind) -> AI coach answer

# --- 2. SIDEBAR ---
llm_configured = False
with st.sidebar:
    try:
        health = api_client.get_health()
        llm_configured = health["llm_configured"]
        st.success("Backend connected")
        st.caption(f"Embedding model: {health['embedding_model']}")
        st.caption("AI coach: " + ("available" if llm_configured else "off (no GROQ_API_KEY)"))
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
    st.caption("Whether an applicant tracking system can read this resume. Independent of the job.")
    for check in results["ats"]["checks"]:
        st.markdown(f"{STATUS_ICONS[check['status']]} **{check['label']}** — {check['message']}")
        if check["examples"]:
            with st.expander("Examples from your resume"):
                for example in check["examples"]:
                    st.markdown(f"- {example}")


def coached(results: dict, kind: str, label: str, key: str, fetch) -> list[dict] | None:
    """A button that asks the AI coach for advice and remembers the answer for this analysis."""
    cache_key = (results["id"], kind)
    if st.button(label, key=f"{key}_{kind}"):
        with st.spinner("Consulting AI coach..."):
            try:
                st.session_state.coaching[cache_key] = fetch()
            except BackendError as e:
                st.error(str(e))
    return st.session_state.coaching.get(cache_key)


def show_learning_path(results: dict, key: str) -> None:
    st.subheader("🎓 Learning Path")
    missing_skills = [s["name"] for s in results["skills"] if s["status"] == "missing"]
    if not missing_skills:
        st.success("No missing skills to plan for.")
        return

    st.caption("A resource and a project idea for the skills this job asks for and you lack.")
    recs = coached(
        results,
        "learning_path",
        "Generate Learning Path",
        key,
        lambda: api_client.get_learning_path(missing_skills, results["job_title"]),
    )
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


def flagged_statements(results: dict) -> list[str]:
    """Resume statements that an ATS check quoted as needing work, without repeats."""
    examples = [
        example
        for check in results["ats"]["checks"]
        if check["key"] in REWRITABLE_CHECKS
        for example in check["examples"]
    ]
    return list(dict.fromkeys(examples))


def show_rewrites(results: dict, key: str) -> None:
    st.subheader("✍️ Statement Rewrites")
    statements = flagged_statements(results)
    if not statements:
        st.success("The ATS check flagged no statements to rewrite.")
        return

    st.caption(
        "Stronger wordings for statements the ATS check flagged. Text in [brackets] is a "
        "placeholder for a real figure; the coach is told never to invent one."
    )
    rewrites = coached(
        results,
        "rewrites",
        "Suggest Rewrites",
        key,
        lambda: api_client.get_rewrites(statements, results["job_title"]),
    )
    for rewrite in rewrites or []:
        with st.container(border=True):
            st.markdown(f"**Before:** {rewrite['original']}")
            st.markdown(f"**After:** {rewrite['rewritten']}")
            st.caption(rewrite["note"])
    if rewrites is not None and not rewrites:
        st.warning("No rewrites returned.")


def show_coach(results: dict, key: str) -> None:
    if not llm_configured:
        st.info("The AI coach is off. Set GROQ_API_KEY on the backend to enable it.")
        return
    show_learning_path(results, key)
    st.divider()
    show_rewrites(results, key)

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

    tab_skills, tab_requirements, tab_ats, tab_coach = st.tabs(
        ["🧩 Skills", "🔍 Requirements", "📋 ATS check", "🎓 AI coach"]
    )
    with tab_skills:
        show_skills(results)
    with tab_requirements:
        show_requirements(results)
    with tab_ats:
        show_ats(results)
    with tab_coach:
        show_coach(results, key)

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
