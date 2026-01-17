import streamlit as st
import os
from data.src.parser import extract_text_from_pdf, extract_text_from_docx
from data.src.cleaner import clean_text
from data.src.analyzer import analyze_resume
from data.src.recommendations import get_ai_recommendations

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("🤖 Phase 4: AI-Powered Resume Analyzer")

# --- 1. SESSION STATE ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- 2. SIDEBAR ---
with st.sidebar:
    st.info("AI Logic: SBERT + Gemini 2.0 / 1.5 Flash")

# --- 3. INPUTS ---
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
with col2:
    jd_input = st.text_area("Paste JD", height=200)

# --- 4. ANALYZE BUTTON ---
if st.button("Analyze Match") and uploaded_file and jd_input:
    with st.spinner("Analyzing..."):
        # Parse
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_docx(uploaded_file)
            
        # Clean
        clean_resume = clean_text(text)
        clean_jd = clean_text(jd_input)
        
        # Analyze & Save
        st.session_state.analysis_results = analyze_resume(text, jd_input, clean_resume, clean_jd)

# --- 5. DISPLAY RESULTS ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    st.divider()
    
    # Score Section
    score = results['match_score']
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
        if results.get('top_sentences'):
            for sentence, val in results['top_sentences']:
                st.info(f"**({int(val*100)}%)** ...{sentence}...")
        else:
            st.info("No strong sentences found.")

    # Missing Keywords
    st.divider()
    st.subheader("⚠️ Missing Keywords")
    if results['missing_keywords']:
        st.write(", ".join([f"`{w}`" for w in results['missing_keywords']]))
        
        # --- GEN AI SECTION ---
        st.divider()
        st.subheader("🎓 AI Personal Growth Plan")
        
        # API Key Logic (Docker Env or Manual Input)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter Gemini API Key:", type="password")
            
        # ... inside the Generate Learning Path button ...
        if st.button("Generate Learning Path"):
            if not api_key:
                st.error("Please enter an API Key.")
            else:
                # REPLACED SPINNER WITH A STATIC MESSAGE
                status_msg = st.empty()
                status_msg.info("⏳ Consulting AI Coach... Please wait.")
                
                # Run the AI
                recs = get_ai_recommendations(results['missing_keywords'], api_key)
                
                # Clear the message once done
                status_msg.empty()
                if isinstance(recs, str):
                        st.error(recs)
                elif recs:
                    cols = st.columns(3)
                    for i, item in enumerate(recs):
                        with cols[i % 3]:
                            with st.container(border=True):
                                st.markdown(f"**📘 Learn {item['skill']}**")
                                    
                                    # --- FIXED KEY ERROR HERE ---
                                    # We use .get() to avoid crashing if a key is missing
                                res_name = item.get('resource_name', 'Recommended Course')
                                link = item.get('link', '#')
                                project = item.get('project', 'No project generated')
                                    
                                st.caption(f"🔗 [{res_name}]({link})")
                                st.success(f"🛠 **Project:** {project}")
                else:
                    st.warning("No recommendations returned.")