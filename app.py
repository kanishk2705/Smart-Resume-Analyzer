# app.py
import streamlit as st
from data.src.parser import extract_text_from_pdf, extract_text_from_docx
from data.src.cleaner import clean_text
from data.src.analyzer import analyze_resume

# Page Config
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

st.title("🤖 Phase 2: AI-Powered Resume Analyzer")
st.markdown("Now using **SBERT Transformers** for semantic understanding.")

# --- SIDEBAR (Optional but clean) ---
with st.sidebar:
    st.header("How it works")
    st.info("1. **SBERT Model** reads the meaning of your resume.\n2. **Gap Analysis** finds missing keywords.\n3. **Hero Logic** highlights your best sentences.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])

with col2:
    st.subheader("2. Job Description")
    jd_input = st.text_area("Paste JD here", height=200)

if st.button("Analyze Match") and uploaded_file and jd_input:
    
    with st.spinner("Reading and encoding vectors..."):
        # 1. Parse Text
        if uploaded_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)
            
        # 2. Clean Text (We need both Raw and Clean now)
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(jd_input)
        
        # 3. Analyze
        results = analyze_resume(resume_text, jd_input, clean_resume, clean_jd)
        
        # --- DISPLAY RESULTS ---
        st.divider()
        
        # Score Section
        col_score, col_details = st.columns([1, 2])
        
        with col_score:
            score = results['match_score']
            st.metric("Match Confidence", f"{score}%")
            if score > 75:
                st.success("High Semantic Match!")
            elif score > 50:
                st.warning("Moderate Match")
            else:
                st.error("Low Match")
                
        with col_details:
            st.subheader("✨ Hero Sentences")
            st.caption("These lines from your resume contributed most to your score:")
            for sentence, score in results['top_sentences']:
                st.info(f"**({int(score*100)}%)** ...{sentence}...")

        # Missing Keywords Section
        st.divider()
        st.subheader("⚠️ Missing Keywords")
        if results['missing_keywords']:
            st.write("Consider adding these exact terms:")
            st.write(", ".join([f"`{w}`" for w in results['missing_keywords']]))
        else:
            st.success("No critical keywords missing!")