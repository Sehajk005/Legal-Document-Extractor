from src.pipeline import process_pdf_for_text
from transformers import pipeline
model = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
threshold = 0.6

def is_legal_document(file_path, session_id):
    full_text = process_pdf_for_text(file_path, session_id)
    input_text = full_text[:1024]
    
    labels = ["legal contract", "resume", "general article", "invoice"]
    
    result = model(input_text, labels=labels)
    
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    
    if(top_score > threshold and top_label == "legal contract"):
        return True, f"Accepted: The top label is {top_label} with a score of {top_score}."
    else:
        return False, f"Rejected: The top label is {top_label} with a score of {top_score}."