"""PDF/Word document parsing. Supports Chinese and English mixed documents."""

import io
import fitz  # PyMuPDF
import docx


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes, one page at a time."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)


def parse_docx(file_bytes: bytes) -> str:
    """Extract text from Word document bytes, one paragraph at a time."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def parse_document(file_bytes: bytes, filename: str) -> str:
    """Parse document by filename extension. Returns full plain text."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return parse_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
