# src/analyzer.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def analyze_resume(resume_text, jd_text):
    # 1. Create a corpus
    corpus = [resume_text, jd_text]

    # 2. Vectorize (Convert text to numbers)
    # We use CountVectorizer here specifically to extract feature names easily for Gap Analysis
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(corpus)

    # 3. Calculate Cosine Similarity
    # This gives a match percentage based on vector alignment
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100

    # 4. Gap Analysis (The "Pro" Feature)
    # Get all unique words (features) found in both documents
    feature_names = cv.get_feature_names_out()

    # Convert sparse matrix to dense array for easier handling
    dense_matrix = count_matrix.todense()
    
    # Create a DataFrame to compare visually
    df = pd.DataFrame(dense_matrix, columns=feature_names, index=['Resume', 'JD'])

    # Find keywords present in JD but 0 in Resume
    # Transpose so rows are keywords, columns are 'Resume' and 'JD'
    df_t = df.T
    
    # Filter: Count in JD > 0 AND Count in Resume == 0
    missing_keywords = df_t[(df_t['JD'] > 0) & (df_t['Resume'] == 0)].index.tolist()

    return {
        "match_score": round(match_percentage, 2),
        "missing_keywords": missing_keywords
    }