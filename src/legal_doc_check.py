import os
import re
from src.pipeline import process_pdf_for_text
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
judge_model = pipeline("text2text-generation", model="google/flan-t5-base")

def verify_document(input_data):
    if os.path.exists(input_data) and input_data.lower().endswith(".pdf"):
        print(f"Processing file: {input_data}")
        full_text = process_pdf_for_text(input_data)
    else:
        full_text = input_data

    text_preview = full_text[:2000]
    
    strong_signals = [
        r"\bhereinafter\b",
        r"\bwitnesseth\b",
        r"in\s+witness\s+whereof",
        r"\bnotwithstanding\b",
        r"\bindemnif(y|ied|ication)\b",
        r"\bseverability\b",
        r"\bforce\s+majeure\b",
        r"\bgoverning\s+law\b",
        r"\bsuccessors\s+and\s+assigns\b",
        r"\bwhereas\b",
        r"\bin\s+consideration\s+of\b",
        r"\bpursuant\s+to\b"
    ]
    
    signal_score = 0
    
    for signal in strong_signals:
        if re.search(signal, text_preview, re.IGNORECASE):
            signal_score = min(signal_score + 0.1, 0.4)
            
    labels = [
        "legal binding contract", 
        "resume or cv", 
        "news report", 
        "financial bill or invoice",
        "general text"
    ]
    
    result = classifier(
        text_preview, 
        labels, 
        hypothesis_template="This document is a {}."
    )
    
    top_label = result['labels'][0]
    top_score = min(result['scores'][0], 0.6)
    
    total_score = signal_score + top_score
    
    return total_score, top_label, text_preview
    
    
    
def is_legal_document(input_data, session_id=None):
    score, top_label, text = verify_document(input_data)
    if(score > 0.6 and top_label != "legal binding contract"):
        return False, f"Rejected: Identified as {top_label} (Score: {score:.2f})"
    if(score >= 0.7 and top_label == "legal binding contract"):
        return True, f"Accepted: {top_label} (Score: {score:.2f})"
    elif(score <= 0.4 and top_label != "legal binding contract"):
        return False, f"Rejected: {top_label} (Score: {score:.2f})"
    elif(0.4 < score < 0.7):
        print(f"--- GRAY ZONE (Score {score:.2f}) - Calling Judge ---")
        prompt = f"""You are a legal document classifier. Analyze the following document. 
        Does this look like a valid legal agreement or contract? Respond with ONLY 'YES' or 'NO'.
        **Document**: {text[:500]}
        Answer:"""
        
        
        
        response = judge_model(prompt)
        generated_text = response[0]['generated_text'].lower()
        
        if "YES" in generated_text:
            return True, f"Accepted by Judge: {top_label} (Original Score: {score:.2f})"
        else:
            return False, f"Rejected by Judge: {top_label} (Original Score: {score:.2f})"