import tempfile
import os
from src.ocr_processing.pdf_processor import convert_pdf_to_images
from src.ocr_processing.image_to_text import extract_text_from_image
from src.cosdata_store import index_document  # <-- NEW: Import our Cosdata indexer

def process_pdf_for_text(file_path):
    with tempfile.TemporaryDirectory() as tmp_dir: # create a temporary directory
        images = convert_pdf_to_images(file_path, tmp_dir)
        full_text = ""
        for filename in os.listdir(tmp_dir):
            full_path = os.path.join(tmp_dir, filename)
            full_text += extract_text_from_image(full_path)
        
        # --- NEW RAG STEP ---
        # After building the full_text, index it in Cosdata.
        # We use the original filename (if possible) or a temp name as the doc_name.
        try:
            doc_name = os.path.basename(file_path)
            print(f"Indexing document: {doc_name} in Cosdata...")
            index_document(full_text, doc_name=doc_name)
            print("Indexing complete.")
        except Exception as e:
            print(f"Error during indexing: {e}")
        # --------------------

        return full_text