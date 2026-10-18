import io

import pytest

from core.parsing import (
    ResumeParseError,
    UnsupportedFileTypeError,
    extract_document,
    extract_text,
)
from tests.conftest import make_docx, make_pdf


def test_extracts_text_from_docx():
    content = make_docx("Built dashboards in Python.", "Managed a SQL warehouse.")

    text = extract_text(io.BytesIO(content), "resume.docx")

    assert text == "Built dashboards in Python.\nManaged a SQL warehouse."


def test_extracts_text_from_pdf():
    text = extract_text(io.BytesIO(make_pdf("Built dashboards in Python")), "resume.pdf")

    assert "Built dashboards in Python" in text


def test_pdf_reports_page_count_and_docx_does_not():
    pdf = extract_document(io.BytesIO(make_pdf("Built dashboards in Python")), "resume.pdf")
    docx = extract_document(io.BytesIO(make_docx("Built dashboards")), "resume.docx")

    assert pdf.page_count == 1
    assert docx.page_count is None


def test_hyperlink_targets_are_extracted():
    pdf = extract_document(
        io.BytesIO(make_pdf("Email", link="mailto:jane@example.com")), "resume.pdf"
    )
    docx = extract_document(
        io.BytesIO(make_docx("Email", link="mailto:jane@example.com")), "resume.docx"
    )

    assert pdf.links == ["mailto:jane@example.com"]
    assert docx.links == ["mailto:jane@example.com"]


def test_docx_tables_are_read_in_document_order():
    content = make_docx("Skills", table=[["Python", "SQL"], ["Docker", "Git"]])

    text = extract_text(io.BytesIO(content), "resume.docx")

    assert text == "Skills\nPython\nSQL\nDocker\nGit"


def test_extension_check_is_case_insensitive():
    text = extract_text(io.BytesIO(make_docx("Python developer")), "RESUME.DOCX")

    assert text == "Python developer"


@pytest.mark.parametrize("filename", ["resume.txt", "resume", "resume.pdf.exe"])
def test_rejects_unsupported_file_types(filename):
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(io.BytesIO(b"anything"), filename)


@pytest.mark.parametrize("filename", ["resume.pdf", "resume.docx"])
def test_corrupted_file_raises_instead_of_returning_the_error_as_text(filename):
    with pytest.raises(ResumeParseError, match="Could not read"):
        extract_text(io.BytesIO(b"this is not a real document"), filename)


def test_file_without_text_raises():
    with pytest.raises(ResumeParseError, match="No text found"):
        extract_text(io.BytesIO(make_docx("   ")), "resume.docx")
