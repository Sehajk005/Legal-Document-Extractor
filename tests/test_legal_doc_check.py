import sys
import os

# Add the project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.legal_doc_check import is_legal_document

def run_test(name, input_text, expected_outcome):
    print(f"--- TEST: {name} ---")
    is_legal, reason = is_legal_document(input_text)
    
    if is_legal == expected_outcome:
        print(f"✅ PASSED. Output: {reason}")
        return True
    else:
        print(f"❌ FAILED. Expected {expected_outcome}, got {is_legal}.")
        print(f"   Reason from Model: {reason}")
        print(f"   Input Snippet: {input_text[:100]}...")
        return False

# --- THE DATASET OF TRAPS ---

# 1. THE HAPPY PATH (Should Pass)
legal_text = """
CONFIDENTIALITY AGREEMENT
This Confidentiality Agreement (the "Agreement") is entered into as of January 1, 2024, by and between 
TechCorp Inc. ("Disclosing Party") and John Doe ("Receiving Party"). 
WHEREAS, the Disclosing Party possesses certain proprietary information...
NOW, THEREFORE, in consideration of the mutual covenants contained herein...
"""

# 2. THE RESUME TRAP (Should Fail)
# Resumes use formal language, dates, and names. Models often confuse them with contracts.
resume_text = """
JOHN DOE
123 Main St, Anytown, USA | john.doe@email.com
EXPERIENCE
Senior Software Engineer, TechCorp (2020-Present)
- Negotiated vendor contracts and managed SLAs.
- Responsible for legal compliance in data handling.
EDUCATION
B.S. Computer Science, University of State
"""

# 3. THE NEWS ARTICLE TRAP (Should Fail)
# It discusses law, courts, and judges, but it is NOT a contract.
news_text = """
SUPREME COURT RULES ON CONTRACT DISPUTE
In a landmark decision today, the Supreme Court ruled that implied covenants of good faith 
are mandatory in all commercial contracts. The judge stated that "Agreement is the foundation of commerce."
Lawyers from both sides argued that the breach of contract was significant.
"""

# 4. THE INVOICE TRAP (Should Fail)
# It has money, dates, and "Terms," but it is a bill, not a legal agreement.
invoice_text = """
INVOICE #10234
Date: 01/01/2024
Bill To: Jane Smith
TERMS: Net 30.
Description: Web Development Services.
Total: $5,000.00
Please remit payment to the account below.
"""

# 5. THE "GIBBERISH" EDGE CASE (Should Fail)
short_text = "This is just a random short sentence that means nothing."

# --- RUNNING THE GAUNTLET ---
results = [
    run_test("Standard Contract", legal_text, True),
    run_test("Resume (The Trap)", resume_text, False),
    run_test("News Article (Context Trap)", news_text, False),
    run_test("Invoice (Financial Trap)", invoice_text, False),
    run_test("Short Text", short_text, False)
]

if all(results):
    print("\n🏆 ALL SYSTEMS GO. The Gatekeeper is ready.")
else:
    print("\n💀 SYSTEM FAILURE. Adjust the threshold or change the model.")