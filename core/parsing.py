import logging
from pathlib import Path
from typing import BinaryIO

import pypdf
from docx import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


class UnsupportedFileTypeError(ValueError):
    """The file extension is not one we can parse."""


class ResumeParseError(ValueError):
    """The file could not be read, or contained no extractable text."""


def extract_text_from_pdf(file: BinaryIO) -> str:
    """Extracts text from a PDF file object."""
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle None returns
    return text


def extract_text_from_docx(file: BinaryIO) -> str:
    """Extracts text from a DOCX file object."""
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text(file: BinaryIO, filename: str) -> str:
    """Extracts text from a resume, choosing the parser from the file extension."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or filename}'. "
            f"Use one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    try:
        if extension == ".pdf":
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_docx(file)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", filename, e)
        raise ResumeParseError(f"Could not read '{filename}'. The file may be corrupted.") from e

    if not text.strip():
        raise ResumeParseError(
            f"No text found in '{filename}'. Scanned or image-only files are not supported."
        )
    return text
