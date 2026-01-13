# src/parser.py
import pypdf
from docx import Document

def extract_text_from_pdf(file):
    """Extracts text from a PDF file object."""
    try:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or "" # Handle None returns
        return text
    except Exception as e:
        return str(e)

def extract_text_from_docx(file):
    """Extracts text from a DOCX file object."""
    try:
        doc = Document(file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return '\n'.join(text)
    except Exception as e:
        return str(e)