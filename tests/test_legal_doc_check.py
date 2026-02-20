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
VENDOR SERVICES AGREEMENT 
This Agreement is made on October 28, 2025 ("Effective Date"). 
BETWEEN: TechCorp Solutions Pvt. Ltd., a company registered in India, with its primary office at 45-B, Elante Mall, Industrial Area Phase I, chandigafh, 160002 (Hereinafter "Company"). 
Contact: priya.singh@techcorp.io
AND: Mr. Rohan Gupta, an independent vendor, residing at House No. 789, Sector 10-C, Chandigarh, 160011 (Hereinafter "Vendor"). 
Contact Phone: +91-9876543210 
WHEREAS:The Company requires services for IT support and maintenance. The Vendor, Mr. Rohan Gupta, has expertise in these areas. 
1. Scope of Work
The Vendor will provide on-call IT support. All work is authorized by Ms. Priya Singh 
(priya.singh@techcorp.io). The Vendor's secondary contact is **O5O-123-4567**. 
2. Payment 
The Company will pay the Vendor a monthly retainer of INR 2,20,000. All invoices must be 
sent to **accounts@techcorp.io**. 
3. Affiliation & Standards
The Vendor agrees to maintain all certifications required by the **Punjab Tech Association** 
(Hereinafter "PTA") for the duration of this contract. The Vendor will also participate in the 
annual audit conducted by the **Chandigarh Legal Society**. 
4. Term and Termination 
This agreement shall last for one (1) year, ending October 27, 2026.
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
    run_test("News Article", news_text, True),
    run_test("Invoice (Financial Trap)", invoice_text, False),
    run_test("Short Text", short_text, False)
]

if all(results):
    print("\n🏆 ALL SYSTEMS GO. The Gatekeeper is ready.")
else:
    print("\n💀 SYSTEM FAILURE. Adjust the threshold or change the model.")