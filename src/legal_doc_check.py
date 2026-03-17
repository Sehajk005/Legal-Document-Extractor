import os
from pydoc import text
import re
from src.pipeline import process_pdf_for_text
_classifier = None
_judge_model = None

def get_classifier():
    global _classifier
    if _classifier is None:
        # Load the zero-shot classification model
        from transformers import pipeline
        _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _classifier

def get_judge_model():
    global _judge_model
    if _judge_model is None:
        # Load the text generation model
        from transformers import pipeline
        _judge_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return _judge_model

negative_patterns = {
    "resume_cv": r'\b(curriculum vitae|references available|linkedin\.com/in/|github\.com/|career objective|extracurricular activities|gpa|cgpa|cumulative grade|bachelor of|master of|seeking a position|internship at)\b',
    
    "personal_email": r'\b(hope this email finds you|just checking in|unsubscribe|view in browser|lol|lmao|promo code|dear friend|miss you|catch up soon|how have you been)\b',
    
    "invoice_receipt": r'\b(ship to|shipping address|cart subtotal|cvv|credit card number|add to cart|checkout|tracking number|invoice no|bill to|payment due|total amount due)\b',
    
    "business_memo": r'\b(key performance indicators|kpi|synergy|all hands meeting|quarterly earnings|go-to-market|action items|next steps|as per our discussion|please find attached|best regards|kind regards|cc:|bcc:)\b',
    
    "general_article": r"\b(celebrity gossip|you won't believe|clickbait|sponsored post|buy now|limited time offer|leave a reply|read more|subscribe now|share this article|views expressed)\b",
}
strong_signals = [
    # --- 1. CONTRACTS & AGREEMENTS (Your existing list + refinements) ---
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
    
    # --- 2. COURT SUMMONS, FINDINGS & LITIGATION ---
    r"\b(plaintiff|defendant|petitioner|respondent)\b",
    r"\b(in\s+the\s+court\s+of|high\s+court|supreme\s+court|district\s+court)\b",
    r"\b(writ\s+petition|civil\s+appeal|criminal\s+appeal)\b",
    r"\b(affidavit|deponent|sworn\s+before|notary\s+public)\b",
    r"\b(cause\s+title|order\s+dated|judgment|decree)\b",
    r"\bsummons\s+to\b",
    r"\b(learned\s+counsel|amicus\s+curiae|stare\s+decisis)\b",
    # --- 3. POLICE COMPLAINTS & FIRs ---
    r"\b(first\s+information\s+report|fir\s+no\.?)\b",
    r"\b(police\s+station|p\.?s\.?)\b",
    r"\b(complainant|accused|informant)\b",
    r"\bunder\s+section\s+\d+[a-z]?\b", # e.g., "under section 420"
    r"\bu/?s\s+\d+[a-z]?\b",            # shorthand "u/s 420"
    r"\b(ipc|crpc|penal\s+code)\b",
    # --- 4. LEGAL NOTICES & DISPUTE EMAILS ---
    r"\b(legal\s+notice|demand\s+letter|cease\s+and\s+desist)\b",
    r"\bwithout\s+prejudice\b",
    r"\b(cause\s+of\s+action|institute\s+legal\s+proceedings)\b",
    r"\bstipulated\s+time\b",
    r"\b(attorney-client\s+privilege|privileged\s+and\s+confidential)\b",
    r"\bbreach\s+of\s+trust\b",
    # --- 5. PROPERTY & CIVIL RECORDS ---
    r"\b(sale\s+deed|title\s+deed|conveyance\s+deed|lease\s+deed)\b",
    r"\b(encumbrance|stamp\s+duty|registration\s+act)\b",
    r"\b(schedule\s+of\s+property|bounded\s+on\s+the)\b",
    r"\b(khasra|khatauni|khatiyan|patta)\b", # Regional land record terms
    
    # --- 6. WILLS & FAMILY LEGAL DOCS ---
    r"\b(last\s+will\s+and\s+testament|testator|testatrix)\b",
    r"\b(bequeath|probate|executor\s+of)\b",
    r"\b(of\s+sound\s+mind|legal\s+heirs)\b",
    # --- 7. TEXTBOOKS & ARTICLES ---
    r"\b(jurisprudence|ratio\s+decidendi|fundamental\s+rights|tort\s+law)\b"
]

def is_negative_pattern(text):
    hits={}
    for name, pat in negative_patterns.items():
        matches = len(re.findall(pat, text, re.IGNORECASE))
        if matches: 
            hits[name] = matches
    total_matches = sum(hits.values())
    dominant_hits = max(hits.values(), default=0)
    
    return total_matches, dominant_hits

def find_signal_positions(text: str) -> list[int]:
    positions = []
    for signal in strong_signals:
        for match in re.finditer(signal, text, re.IGNORECASE):
            positions.append(match.start())
    return sorted(positions)

def extract_densest_chunks(text: str, chunk_size: int = 1500) -> str:
    positions = find_signal_positions(text)
    
    if not positions:
        start = len(text) // 4
        return text[start:start + chunk_size]
    
    # Find the densest chunk around the strong signals
    best_start = 0
    best_count = 0
    
    for pos in positions:
        window_start = max(0, pos - 200)
        window_end = window_start + chunk_size
        
        hits_in_window = sum(1 for p in positions if window_start <= p < window_end)
        
        if hits_in_window > best_count:
            best_count = hits_in_window
            best_start = window_start
    
    return text[best_start:best_start + chunk_size]

def extract_judge_context(text: str, chunk_size: int = 1500) -> str:
    chunk = extract_densest_chunks(text, chunk_size)
    
    # Clean up whitespace artifacts from PDF extraction
    chunk = re.sub(r'\n{3,}', '\n\n', chunk) 
    chunk = re.sub(r'[ \t]{2,}', ' ', chunk) 
    chunk = chunk.strip()

    return chunk

def verify_document(input_data):
    if os.path.exists(input_data) and input_data.lower().endswith(".pdf"):
        print(f"Processing file: {input_data}")
        full_text = process_pdf_for_text(input_data)
    else:
        full_text = input_data
        
    chunks = extract_judge_context(full_text)
    
    
    signal_score = 0
    
    for signal in strong_signals:
        if re.search(signal, chunks, re.IGNORECASE):
            signal_score = min(signal_score + 0.05, 0.2)
            
    labels = [
        # ACCEPT
        "a legal document",
        
        # REJECT
        "not a legal document"
    ]
    
    
    
    classifier = get_classifier()
    result = classifier(
        chunks, 
        labels, 
        hypothesis_template="This document is a {}."
    )
    
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    

    if top_label == "a legal document":
        total_score = min(top_score + signal_score, 1.0)
    else:
        total_score = max(top_score - (signal_score * 0.5), 0.0)
    
    # debugging
    # print(f"[DEBUG] signal={signal_score:.2f} | zsc={top_score:.2f} | total={total_score:.2f} | label={top_label}")
    
    return total_score, top_label, full_text, top_score, chunks
    
    
    
def is_legal_document(input_data):
    score, top_label, text, classifier_score, chunks = verify_document(input_data)
    total_matches, dominant_hits = is_negative_pattern(text)
    
    
    min_classifier_score = 0.6
    
    score = round(score, 2)
    min_classifier_score = round(min_classifier_score, 2)
    
    if total_matches >= 4 or dominant_hits >= 3:
        return False, chunks, "Rejected: Too many negative patterns." 
    
    if(classifier_score <= min_classifier_score):
        return False, chunks, f"Rejected: Classifier confidence too low ({classifier_score:.2f})."
    
    if(score >= 0.8 and top_label != "a legal document"):
        return False, chunks, f"Rejected: Identified as {top_label} (Score: {score:.2f})"
    
    if(score >= 0.8 and top_label == "a legal document"):
        return True, chunks, f"Accepted: {top_label} (Score: {score:.2f}) Chunks: {chunks}"
    
    elif(score <= 0.4):
        return False, chunks, f"Rejected: {top_label} (Score: {score:.2f}) "
    
    elif(0.4 < score < 0.8):
        print(f"--- GRAY ZONE (Score {score:.2f}) - Calling Judge ---")
        prompt = f"""You are a legal document classifier for a legal AI application.
        Classify the document below. Answer only "yes" or "no".
        Answer "yes" if the document is legal document (e.g., contract, agreement, court order, legal notice, affidavit).

        Answer "no" if the document is not a legal document (e.g., resume, personal email, invoice, business memo, general article).

        Document: {chunks}

        Is this a legal document? Answer yes or no:"""
        
        judge_model = get_judge_model()
        response = judge_model(prompt)
        generated_text = response[0]['generated_text'].lower()
        
        if "yes" in generated_text:
            return True, chunks, f"Accepted by Judge: {top_label} (Original Score: {score:.2f})"
        else:
            return False, chunks, f"Rejected by Judge: {top_label} (Original Score: {score:.2f})"