import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pypdf
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.table import Table

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


class UnsupportedFileTypeError(ValueError):
    """The file extension is not one we can parse."""


class ResumeParseError(ValueError):
    """The file could not be read, or contained no extractable text."""


@dataclass
class ParsedDocument:
    text: str
    # Hyperlink targets, which often hold the email or LinkedIn URL behind an icon or label
    links: list[str] = field(default_factory=list)
    page_count: int | None = None  # None when the format has no fixed pages (DOCX)


def _pdf_links(page) -> list[str]:
    links = []
    for annotation in page.get("/Annots") or []:
        action = annotation.get_object().get("/A")
        if action and "/URI" in action:
            links.append(str(action["/URI"]))
    return links


def extract_from_pdf(file: BinaryIO) -> ParsedDocument:
    """Extracts text and hyperlinks from a PDF file object."""
    pdf_reader = pypdf.PdfReader(file)
    pages, links = [], []
    for page in pdf_reader.pages:
        pages.append(page.extract_text() or "")  # Handle None returns
        try:
            links.extend(_pdf_links(page))
        except Exception as e:  # Links are a bonus; malformed annotations must not fail parsing
            logger.debug("Could not read PDF links: %s", e)
    return ParsedDocument("\n".join(pages), links, len(pdf_reader.pages))


def extract_from_docx(file: BinaryIO) -> ParsedDocument:
    """Extracts text and hyperlinks from a DOCX file object."""
    doc = Document(file)
    links = [
        rel.target_ref
        for rel in doc.part.rels.values()
        if rel.reltype == RELATIONSHIP_TYPE.HYPERLINK
    ]
    return ParsedDocument(_docx_text(doc), links)


def _docx_text(doc) -> str:
    lines = []
    # Paragraphs and tables in document order; many resume templates are laid out as tables
    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            seen = set()
            for row in block.rows:
                for cell in row.cells:
                    # A merged cell is reported once per column it spans
                    if cell._tc not in seen:
                        seen.add(cell._tc)
                        lines.append(cell.text)
        else:
            lines.append(block.text)
    return "\n".join(lines)


def extract_text(file: BinaryIO, filename: str) -> str:
    """Extracts just the text from a resume."""
    return extract_document(file, filename).text


def extract_document(file: BinaryIO, filename: str) -> ParsedDocument:
    """Extracts a resume, choosing the parser from the file extension."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or filename}'. "
            f"Use one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    try:
        document = extract_from_pdf(file) if extension == ".pdf" else extract_from_docx(file)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", filename, e)
        raise ResumeParseError(f"Could not read '{filename}'. The file may be corrupted.") from e

    if not document.text.strip():
        raise ResumeParseError(
            f"No text found in '{filename}'. Scanned or image-only files are not supported."
        )
    return document
