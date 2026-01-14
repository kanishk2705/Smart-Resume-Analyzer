# src/analyzer.py
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk

# Ensure we have the sentence tokenizer
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load the model globally (so it doesn't reload every time you click button)
print("Loading Transformer Model... (This runs once)")
model = SentenceTransformer('all-MiniLM-L6-v2')

def analyze_resume(resume_text_raw, jd_text_raw, clean_resume, clean_jd):
    """
    Args:
        resume_text_raw: Original text (for SBERT context)
        jd_text_raw: Original JD (for SBERT context)
        clean_resume: Processed text (for Keyword Gap Analysis)
        clean_jd: Processed JD (for Keyword Gap Analysis)
    """
    
    # --- PART 1: OVERALL SEMANTIC SCORE (The "Smart" Score) ---
    # We use the RAW text because BERT needs grammar to understand context
    embeddings1 = model.encode(resume_text_raw, convert_to_tensor=True)
    embeddings2 = model.encode(jd_text_raw, convert_to_tensor=True)
    
    # Calculate Cosine Similarity of the meanings
    similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    match_percentage = round(similarity_score * 100, 2)


    # --- PART 2: HERO SENTENCE EXTRACTION (Explainability) ---
    # "Why did I get this score?" -> We identify the best matching sentences
    sentences = nltk.sent_tokenize(resume_text_raw)
    
    # Filter short garbage sentences (e.g., "Page 1", "Skills")
    valid_sentences = [s for s in sentences if len(s.split()) > 4]
    
    top_sentences = []
    if valid_sentences:
        # Encode all resume sentences
        sent_embeddings = model.encode(valid_sentences, convert_to_tensor=True)
        
        # Compare every sentence against the JD
        cosine_scores = util.pytorch_cos_sim(sent_embeddings, embeddings2)
        
        # Pair sentence with score
        scored_sentences = []
        for i, score in enumerate(cosine_scores):
            scored_sentences.append((valid_sentences[i], score.item()))
            
        # Sort by highest score first and take top 3
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:3]


    # --- PART 3: KEYWORD GAP ANALYSIS (The "Strict" Check) ---
    # We use the CLEAN text here to match specific skills (e.g., "Python")
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    missing_keywords = []
    
    try:
        count_matrix = cv.fit_transform([clean_resume, clean_jd])
        feature_names = cv.get_feature_names_out()
        
        # Create DataFrame
        df = pd.DataFrame(count_matrix.todense(), columns=feature_names, index=['Resume', 'JD'])
        df_t = df.T
        
        # Find words in JD but NOT in Resume
        missing_df = df_t[(df_t['JD'] > 0) & (df_t['Resume'] == 0)]
        missing_df = missing_df.sort_values(by='JD', ascending=False)
        missing_keywords = missing_df.index.tolist()
        
    except ValueError:
        # Happens if documents are empty or have no shared words
        pass

    return {
        "match_score": match_percentage,
        "missing_keywords": missing_keywords[:10], # Top 10 missing
        "top_sentences": top_sentences  # Returns [(sentence, score), ...]
    }