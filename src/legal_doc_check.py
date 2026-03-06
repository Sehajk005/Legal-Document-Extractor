import os
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

def is_negative_pattern(text):
    hits={}
    for name, pat in negative_patterns.items():
        matches = len(re.findall(pat, text, re.IGNORECASE))
        if matches: 
            hits[name] = matches
    total_matches = sum(hits.values())
    dominant_hits = max(hits.values(), default=0)
    
    return total_matches, dominant_hits

def verify_document(input_data):
    if os.path.exists(input_data) and input_data.lower().endswith(".pdf"):
        print(f"Processing file: {input_data}")
        full_text = process_pdf_for_text(input_data)
    else:
        full_text = input_data

    text_preview = full_text[:2000]
    
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
    
    signal_score = 0
    
    for signal in strong_signals:
        if re.search(signal, text_preview, re.IGNORECASE):
            signal_score = min(signal_score + 0.1, 0.4)
            
    labels = [
        # ACCEPT
        "a legal contract, binding agreement, or personal legal document like a will",
        "a police complaint, FIR, court summons, or judicial court finding",
        "a legal notice, demand letter, or legal email correspondence",
        "a property deed, civil record, or land registry document",
        "a legal textbook, law article, or educational legal blog post",
        
        # REJECT
        "a resume, curriculum vitae, or professional profile",
        "a financial invoice, bill, receipt, or payment document",
        "a news article, blog post, or general informational content",
        "a business email, personal letter, memo, or meeting minutes, a personal friendly email",
    ]
    
    ACCEPT_LABELS = [
        "a legal contract, binding agreement, or personal legal document like a will",
        "a police complaint, FIR, court summons, or judicial court finding",
        "a legal notice, demand letter, or legal email correspondence",
        "a property deed, civil record, or land registry document",
        "a legal textbook, law article, or educational legal blog post",
    ]
    
    classifier = get_classifier()
    result = classifier(
        text_preview, 
        labels, 
        hypothesis_template="This document is a {}."
    )
    
    top_label = result['labels'][0]
    top_score = result['scores'][0] * 0.6
    

    total_score = signal_score + top_score
    
    # debugging
    # print(f"[DEBUG] signal={signal_score:.2f} | zsc={top_score:.2f} | total={total_score:.2f} | label={top_label}")
    
    return total_score, top_label, full_text, ACCEPT_LABELS
    
    
    
def is_legal_document(input_data):
    score, top_label, text, ACCEPT_LABELS = verify_document(input_data)
    total_matches, dominant_hits = is_negative_pattern(text)
    
    if total_matches >= 4 or dominant_hits >= 3:
        return False, "Rejected: Too many negative patterns." 
    
    if(score >= 0.6 and top_label not in ACCEPT_LABELS):
        return False, f"Rejected: Identified as {top_label} (Score: {score:.2f})"
    
    if(score >= 0.7 and top_label in ACCEPT_LABELS):
        return True, f"Accepted: {top_label} (Score: {score:.2f})"
    
    elif(score <= 0.4):
        return False, f"Rejected: {top_label} (Score: {score:.2f})"
    
    elif(0.4 < score < 0.7):
        print(f"--- GRAY ZONE (Score {score:.2f}) - Calling Judge ---")
        prompt = f"""You are a legal document classifier for a legal AI application.
        Classify the document below. Answer only "yes" or "no".
        Answer "yes" if the document is ONE of these:
        - A legal contract, service agreement, NDA, or binding agreement between parties
        - A police complaint, FIR, court summons, judgment, or court order
        - A legal notice, demand letter, or cease and desist letter
        - A property deed, sale deed, land registry, or civil record
        - A legal textbook, law article, or educational content about law

        Answer "no" if the document is ONE of these:
        - A resume, CV, or professional profile listing work experience
        - An invoice, bill, receipt, or payment request
        - A news article, blog post, or general informational article
        - A business email, personal letter, office memo, or meeting minutes

        Document: {text[:500]}

        Is this a legal document? Answer yes or no:"""
        
        judge_model = get_judge_model()
        response = judge_model(prompt)
        generated_text = response[0]['generated_text'].lower()
        
        if "yes" in generated_text:
            return True, f"Accepted by Judge: {top_label} (Original Score: {score:.2f})"
        else:
            return False, f"Rejected by Judge: {top_label} (Original Score: {score:.2f})"