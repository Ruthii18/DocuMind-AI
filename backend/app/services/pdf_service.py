import io
from typing import List, Dict, Any
from pypdf import PdfReader


class PDFExtractionError(Exception):
    """Exception raised when PDF extraction fails."""
    pass


class PDFService:
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF pages, preserving 1-based page numbers.
        Returns a list of dicts: [{"page_number": int, "text": str}]
        """
        if not filename.lower().endswith(".pdf"):
            raise PDFExtractionError("Invalid file type. Only PDF files are supported.")

        if not file_bytes:
            raise PDFExtractionError("Uploaded file is empty.")

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            raise PDFExtractionError(f"Failed to read PDF file: {str(e)}")

        if len(reader.pages) == 0:
            raise PDFExtractionError("The uploaded PDF has no pages.")

        extracted_pages: List[Dict[str, Any]] = []
        total_text_length = 0

        for idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                # Normalize spaces and strip leading/trailing whitespace
                cleaned_text = " ".join(page_text.split())
                if cleaned_text:
                    total_text_length += len(cleaned_text)
                    extracted_pages.append({
                        "page_number": idx,
                        "text": cleaned_text
                    })
                else:
                    # Still record page with empty text to preserve page mapping info
                    extracted_pages.append({
                        "page_number": idx,
                        "text": ""
                    })
            except Exception as e:
                # Log or handle individual page extraction error
                extracted_pages.append({
                    "page_number": idx,
                    "text": ""
                })

        # Check if entire PDF contains no extractable text (e.g. scanned image PDF without OCR)
        if total_text_length == 0:
            raise PDFExtractionError(
                "No selectable text could be extracted from this PDF. "
                "The file may be a scanned image or empty. OCR is not supported in this version."
            )

        return extracted_pages

