import requests
import json
import urllib.parse

def get_ai_recommendations(missing_skills, api_key):
    if not missing_skills or not api_key:
        return None

    # LIST OF MODELS TO TRY
    models_to_try = [
        "gemini-2.5-flash", 
        "gemini-2.0-flash-exp", 
        "gemini-1.5-flash"
    ]
    
    headers = {'Content-Type': 'application/json'}
    last_error = ""

    # UPDATED PROMPT: We ask for a "Search Query" instead of a "Link"
    target_skills = missing_skills[:3]
    skills_str = ", ".join(target_skills)
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""
                You are a career coach. A candidate is missing: {skills_str}.
                For each skill, provide:
                1. A specific resource name (e.g. "Coursera: Python for Everybody").
                2. A project idea.
                
                Return strictly JSON.
                Format: [{{"skill": "...", "resource_name": "...", "project": "..."}}]
                """
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json"
        }
    }

    for model_name in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and data['candidates']:
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    recommendations = json.loads(text)
                    
                    # 🚀 FIX: GENERATE GOOGLE SEARCH LINKS MANUALLY
                    for rec in recommendations:
                        query = urllib.parse.quote(rec['resource_name'])
                        rec['link'] = f"https://www.google.com/search?q={query}"
                        
                    return recommendations
            
            last_error = f"{model_name}: {response.status_code}"
            continue

        except Exception as e:
            last_error = str(e)
            continue

    return f"Error: {last_error}"