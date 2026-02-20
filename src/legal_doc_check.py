import os
import re
from src.pipeline import process_pdf_for_text
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
judge_model = pipeline("text2text-generation", model="google/flan-t5-base")

negative_patterns = {
    "resume_cv": r'\b(curriculum vitae|work experience|education background|references available|linkedin\.com|github\.com|years of experience|proficient in|skills summary|objective statement|career objective)\b',
    
    "personal_email": r'\b(dear\s+\w+|best regards|kind regards|sincerely yours|sent from my iphone|unsubscribe)\b',
    
    "invoice_receipt": r'\b(invoice number|bill to|payment due|subtotal|tax amount|receipt number|purchase order)\b',
    
    "business_memo": r'\b(memo to|memorandum to|prepared by|submitted to|action items|meeting minutes|agenda for)\b',
    
    "general_article": r'\b(trending now|click here|subscribe to|follow us on|share this post|comments section|tags:)\b',
}

def is_negative_pattern(text):
    hits={}
    for name, pat in negative_patterns.items():
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches: 
            hits[name] = matches
    negative_score = min(len(hits) * 0.1, 0.3)
    
    return negative_score

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
        r"\bpursuant\s+to\b",
        r'\bTHIS\s+AGREEMENT\s+(is\s+entered|made|dated|shall)',
        r'\bTHIS\s+[A-Z\s]+AGREEMENT\b',   
        r'^AGREEMENT\s*\n',                
        r'AGREEMENT\s+between\s+.+and\s+',
        r'["\u201c]Agreement["\u201d]\s+means'
    ]
    
    signal_score = 0
    
    for signal in strong_signals:
        if re.search(signal, text_preview, re.IGNORECASE):
            signal_score = min(signal_score + 0.1, 0.4)
            
    labels = [
        # ACCEPT
        "a legal contract, agreement, deed, will, or binding legal instrument",
        "a legal article, case law, statute, regulation, or academic legal writing",
        
        # REJECT
        "a resume, curriculum vitae, or professional profile",
        "a financial invoice, bill, receipt, or payment document",
        "a news article, blog post, or general informational content",
        "a business email, personal letter, memo, or meeting minutes",
    ]
    
    ACCEPT_LABELS = [
        "a legal contract, agreement, deed, will, or binding legal instrument",
        "a legal article, case law, statute, regulation, or academic legal writing",
    ]
    
    result = classifier(
        text_preview, 
        labels, 
        hypothesis_template="This document is a {}."
    )
    
    top_label = result['labels'][0]
    top_score = result['scores'][0] * 0.5
    
    negative_score = is_negative_pattern(text_preview)
    
    total_score = signal_score + top_score - negative_score
    
    # debugging
    # print(f"[DEBUG] signal={signal_score:.2f} | zsc={top_score:.2f} | neg={negative_score:.2f} | total={total_score:.2f} | label={top_label}")
    
    return total_score, top_label, text_preview, ACCEPT_LABELS
    
    
    
def is_legal_document(input_data, session_id=None):
    score, top_label, text, ACCEPT_LABELS = verify_document(input_data)
    
    if(score >= 0.6 and top_label not in ACCEPT_LABELS):
        return False, f"Rejected: Identified as {top_label} (Score: {score:.2f})"
    
    if(score >= 0.7 and top_label in ACCEPT_LABELS):
        return True, f"Accepted: {top_label} (Score: {score:.2f})"
    
    elif(score <= 0.4):
        return False, f"Rejected: {top_label} (Score: {score:.2f})"
    
    elif(0.4 < score < 0.7):
        print(f"--- GRAY ZONE (Score {score:.2f}) - Calling Judge ---")
        prompt = f"""You are a legal document classifier for a legal AI application.
        ACCEPT the following:
        - Contracts, agreements, NDAs, leases, deeds, wills, affidavits
        - Court orders, judgments, legal filings
        - Legal articles, law review papers, legal blogs
        - Law textbooks, statutes, regulations, case law summaries
        - Any document primarily about legal topics, rights, or obligations

        REJECT the following:
        - Resumes or CVs (even if the person is a lawyer)
        - Personal emails or letters
        - Invoices, receipts, or financial statements
        - Business memos or meeting minutes
        - General non-legal articles or marketing content

        Does the following document fall under the ACCEPT category?
        Answer ONLY yes or no.
        
        **Document**: {text[:500]}
        Answer:"""
        
        response = judge_model(prompt)
        generated_text = response[0]['generated_text'].lower()
        
        if "yes" in generated_text:
            return True, f"Accepted by Judge: {top_label} (Original Score: {score:.2f})"
        else:
            return False, f"Rejected by Judge: {top_label} (Original Score: {score:.2f})"