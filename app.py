# app.py
import streamlit as st
from data.src.parser import extract_text_from_pdf, extract_text_from_docx
from data.src.cleaner import clean_text
from data.src.analyzer import analyze_resume

# Page Config
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

st.title("🚀 Smart Resume & Gap Analyzer")
st.markdown("Upload your resume and paste the Job Description (JD) to see your match score and **missing keywords**.")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    
    resume_text = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                resume_text = extract_text_from_docx(uploaded_file)
            
            st.success("Resume uploaded successfully!")
            with st.expander("Show Extracted Resume Text"):
                st.write(resume_text[:500] + "...") # Show preview
        except Exception as e:
            st.error(f"Error reading file: {e}")

with col2:
    st.header("2. Job Description")
    jd_input = st.text_area("Paste the JD here:", height=300)

# The "Action" Button
if st.button("Analyze Match"):
    if resume_text and jd_input:
        
        # 1. Clean Data
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(jd_input)
        
        # 2. Analyze
        results = analyze_resume(clean_resume, clean_jd)
        
        # 3. Display Results
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Score Gauge
        score = results['match_score']
        if score > 75:
            st.balloons()
            st.success(f"**Match Score: {score}%** (Excellent)")
        elif score > 50:
            st.warning(f"**Match Score: {score}%** (Good, but needs work)")
        else:
            st.error(f"**Match Score: {score}%** (Low match)")
            
        # Missing Keywords (The "Pro" Value Add)
        st.subheader("⚠️ Missing Keywords")
        st.write("These words appear in the JD but are missing from your resume:")
        
        missing = results['missing_keywords']
        if missing:
            # Display as tags
            st.write(", ".join([f"`{word}`" for word in missing]))
        else:
            st.info("No major keywords missing!")
            
    else:
        st.warning("Please upload a resume and provide a JD.")