"""Tests for file parser module."""
import pytest
import json
from unittest.mock import patch, MagicMock
from file_parser import extract_text_from_bytes


class TestExtractTextFromBytes:
    def test_extract_text_utf8(self):
        content = "Hello, world!".encode("utf-8")
        result = extract_text_from_bytes(content, "test.txt")
        assert result == "Hello, world!"

    def test_extract_text_latin1(self):
        content = "Café résumé".encode("latin-1")
        result = extract_text_from_bytes(content, "test.txt")
        assert "Café" in result

    def test_extract_json(self):
        data = {"key": "value", "number": 42}
        content = json.dumps(data).encode("utf-8")
        result = extract_text_from_bytes(content, "data.json")
        assert "key" in result

    def test_extract_csv(self):
        content = "name,age\nAlice,30\nBob,25".encode("utf-8")
        result = extract_text_from_bytes(content, "data.csv")
        assert "Alice" in result

    def test_extract_markdown(self):
        content = "# Title\n\nSome content".encode("utf-8")
        result = extract_text_from_bytes(content, "doc.md")
        assert "Title" in result

    def test_unsupported_file_type(self):
        content = b"\x00\x01\x02\x03"
        with pytest.raises(ValueError):
            extract_text_from_bytes(content, "file.xyz")

    def test_fallback_to_text_for_unknown(self):
        content = "plain text content".encode("utf-8")
        result = extract_text_from_bytes(content, "file.unknown")
        assert result == "plain text content"

    def test_image_returns_placeholder(self):
        content = b"\x89PNG\r\n\x1a\n"
        result = extract_text_from_bytes(content, "photo.png")
        assert "Image file" in result
        assert "photo.png" in result

    def test_jpeg_returns_placeholder(self):
        content = b"\xff\xd8\xff"
        result = extract_text_from_bytes(content, "photo.jpg")
        assert "Image file" in result

    def test_empty_text_file(self):
        content = "".encode("utf-8")
        result = extract_text_from_bytes(content, "empty.txt")
        assert result == ""


class TestPDFExtraction:
    def test_pdf_with_text_layer(self):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Invoice from Test Corp", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        result = extract_text_from_bytes(pdf_bytes, "test.pdf")
        assert "Invoice from Test Corp" in result
        assert "Page 1" in result

    def test_pdf_with_multiple_pages(self):
        import fitz
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1} content", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        result = extract_text_from_bytes(pdf_bytes, "multi.pdf")
        assert "Page 1 content" in result
        assert "Page 2 content" in result
        assert "Page 3 content" in result

    def test_scanned_pdf_uses_ocr(self):
        import fitz
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Create a PDF with an image containing text (no text layer)
        img = Image.new("RGB", (800, 200), "white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 80), "Scanned Invoice Data", fill="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        doc = fitz.open()
        page = doc.new_page()
        page.insert_image(page.rect, stream=img_bytes.getvalue())
        pdf_bytes = doc.tobytes()
        doc.close()

        result = extract_text_from_bytes(pdf_bytes, "scanned.pdf")
        # OCR should extract the text from the image
        assert "Scanned" in result or "Invoice" in result or "OCR" in result

    def test_empty_pdf_raises_error(self):
        import fitz
        doc = fitz.open()
        doc.new_page()  # empty page
        pdf_bytes = doc.tobytes()
        doc.close()

        with pytest.raises(ValueError, match="No text content found"):
            extract_text_from_bytes(pdf_bytes, "empty.pdf")

    def test_ocr_fallback_when_no_tesseract(self):
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        with patch.dict("sys.modules", {"pytesseract": None}):
            with pytest.raises(ValueError, match="No text content found"):
                extract_text_from_bytes(pdf_bytes, "no_ocr.pdf")
