"""Multi-format file parser for document text extraction."""
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_tesseract():
    """Configure pytesseract to find tesseract-ocr executable."""
    try:
        import pytesseract
        
        # Check if tesseract is already in PATH
        try:
            pytesseract.get_tesseract_version()
            return  # Already configured
        except Exception:
            pass
        
        # Windows: check common installation paths
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info("Configured tesseract path: %s", path)
                    return
        
        # Linux/Mac: check common paths
        elif platform.system() in ("Linux", "Darwin"):
            possible_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info("Configured tesseract path: %s", path)
                    return
                    
    except ImportError:
        logger.warning("pytesseract not installed")
    except Exception as e:
        logger.warning("Failed to configure tesseract: %s", e)


# Configure tesseract on module load
_configure_tesseract()


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """Extract text content from file bytes.

    Supports:
    - PDF files (.pdf) - uses PyMuPDF (fitz)
    - Plain text (.txt, .json, .csv, .md, .html, .xml)
    - Images (.png, .jpg, .jpeg, .tiff, .bmp) - returns placeholder for OCR

    Args:
        content: Raw file bytes
        filename: Original filename (used to detect type)

    Returns:
        Extracted text content

    Raises:
        ValueError: If file type is unsupported or extraction fails
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(content)
    elif suffix in (".txt", ".json", ".csv", ".md", ".html", ".xml", ".py", ".js"):
        return _extract_text(content)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"):
        return _extract_image(content, filename)
    else:
        # Try text extraction as fallback
        try:
            return _extract_text(content)
        except Exception:
            raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF using PyMuPDF.

    For scanned/image-based PDFs, falls back to OCR via pytesseract.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        # If no text found, try OCR on rendered pages
        if not text_parts:
            logger.info("No text layer found, attempting OCR on %d pages", len(doc))
            text_parts = _ocr_pdf_pages(doc)

        doc.close()

        if not text_parts:
            raise ValueError(
                "No text content found in PDF. The file may be a scanned image "
                "without an OCR text layer. Please ensure OCR is installed "
                "(tesseract-ocr) and try again."
            )

        return "\n\n".join(text_parts)
    except ImportError:
        raise ValueError("PyMuPDF not installed. Run: pip install pymupdf")
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")


def _ocr_pdf_pages(doc) -> list[str]:
    """Render PDF pages to images and OCR them using pytesseract."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to high-res image (300 DPI for good OCR)
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # OCR the image
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                text_parts.append(f"--- Page {page_num + 1} (OCR) ---\n{page_text.strip()}")

        return text_parts
    except ImportError:
        logger.warning("pytesseract not available, cannot OCR scanned PDF")
        return []
    except Exception as e:
        logger.warning("OCR failed for PDF pages: %s", e)
        return []


def _extract_text(content: bytes) -> str:
    """Extract text from plain text files."""
    if not content:
        return ""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            decoded = content.decode(encoding)
            printable_ratio = sum(c.isprintable() or c in "\n\r\t" for c in decoded) / max(len(decoded), 1)
            if printable_ratio >= 0.85:
                return decoded
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("Could not decode text file")


def _extract_image(content: bytes, filename: str) -> str:
    """Handle image files - returns metadata for OCR processing."""
    return f"[Image file: {filename}] OCR text extraction not yet implemented. Upload a text-based PDF or document instead."
