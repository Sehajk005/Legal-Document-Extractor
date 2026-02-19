import tempfile
import os
import fitz  # PyMuPDF
from src.ocr_processing.pdf_processor import convert_pdf_to_images
from src.ocr_processing.image_to_text import extract_text_from_image
# from src.cosdata_store import index_document

MIN_TEXT_LENGTH_FOR_DIGITAL = 100  

def attempt_digital_extraction(file_path):
    """Tries to extract text directly. Returns (text, is_digital)"""
    print("Attempting digital extraction...")
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        if len(full_text.strip()) > MIN_TEXT_LENGTH_FOR_DIGITAL:
            print("Digital extraction SUCCESSFUL.")
            return full_text, True
        else:
            print("Digital extraction failed (text too short), falling back to OCR.")
            return None, False
    except Exception as e:
        print(f"Digital extraction error: {e}. Falling back to OCR.")
        return None, False

def perform_ocr_extraction(file_path, tmp_dir):
    """Your original OCR-based image pipeline."""
    print("Performing full OCR extraction...")
    images = convert_pdf_to_images(file_path, tmp_dir)
    full_text = ""
    for filename in os.listdir(tmp_dir):
        if filename.endswith(".jpg"): 
            full_path = os.path.join(tmp_dir, filename)
            full_text += extract_text_from_image(full_path)
    print("OCR extraction complete.")
    return full_text

def process_pdf_for_text(file_path):
    """
    New pipeline: Hybrid Parsing + Session-aware Indexing.
    """
    # 1. Try fast digital extraction
    full_text, is_digital = attempt_digital_extraction(file_path)

    if not is_digital:
        # 2. Fallback to slow OCR
        with tempfile.TemporaryDirectory() as tmp_dir:
            full_text = perform_ocr_extraction(file_path, tmp_dir)
            
    return full_text