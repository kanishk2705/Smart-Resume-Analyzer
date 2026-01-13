# src/cleaner.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- UPDATED NLTK DOWNLOAD SECTION ---
# We force download 'punkt_tab' to fix the specific error you are facing
def download_nltk_data():
    packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for package in packages:
        try:
            # Check if it exists specifically to avoid re-downloading every time
            if package == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            else:
                nltk.data.find(f'corpora/{package}' if package != 'punkt' else f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading missing NLTK resource: {package}...")
            nltk.download(package)

# Run the download check immediately when this module is imported
download_nltk_data()

def clean_text(text):
    if not text:
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special chars (keep only alphabets)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization (Split into words)
    # This is where your error was happening
    tokens = nltk.word_tokenize(text)
    
    # 4. Initialize Lemmatizer and Stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # 5. The NLP Pipeline Loop
    clean_tokens = []
    for word in tokens:
        if word not in stop_words:
            # Lemmatize: "developing" -> "develop"
            root_word = lemmatizer.lemmatize(word)
            clean_tokens.append(root_word)
            
    # Join back into a string
    return " ".join(clean_tokens)