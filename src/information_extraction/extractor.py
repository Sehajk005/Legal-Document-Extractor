import os
from dotenv import load_dotenv
import google.generativeai as genai
from src.cosdata_store import query_cosdata  # <-- NEW: Import our Cosdata searcher

# This line loads the variables from your .env file
load_dotenv()

# This safely gets the key from the environment
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-flash-latest")


# --- STEP 1: DEFINE RESPONSIBILITY PRINCIPLES ---
# We define our safety rules directly in the code for simplicity.
RESPONSIBILITY_PRINCIPLES = [
    "1. DO NOT PROVIDE LEGAL ADVICE: The answer must not give definitive legal advice, predict legal outcomes, or tell the user what they 'should' do. It must only state what the document says.",
    "2. STICK TO THE CONTEXT: The answer must be 100% based on the provided document context. It must not invent or infer information not present in the snippets.",
    "3. AVOID BIAS: The answer must not use biased language or make assumptions about any party based on demographics.",
    "4. MAINTAIN NEUTRALITY: The answer should be neutral and objective, simply summarizing the facts of the document."
]
# -------------------------------------------------



def extract_entities_with_llm(text):
    full_prompt = f"""You are an expert legal assistant. From the document text provided, perform two tasks:
    
    Task 1: Extract key entities.
    Task 2: Analyze and extract key clauses.

    Return your response as a single, raw JSON object with two top-level keys: "entities" and "clauses".

    The "entities" key should contain an object with the following fields:
    - "individual_names"
    - "dates"
    - "addresses_locations"
    - "phone_numbers"
    - "emails"
    - "company_names"
    - "organization_names"

    The "clauses" key should contain a JSON array where each object in the array represents a clause and has the following fields:
    - "clause_title"
    - "clause_type" (Categorize from: [Termination, Payment, Liability, Confidentiality, Governing Law, Force MajeMajeure, General, Other])
    - "clause_text" (The exact text of the clause)
    - "summary_in_plain_english"
    - "potential_risks"

    IMPORTANT: Your entire output must be only the raw JSON object, starting with {{ and ending with }}. Do not include any other text or markdown formatting.

    Here is the document:
    ---
    {text}
    ---
    """
    response = model.generate_content(full_prompt)
    return response.text

def answer_user_questions(user_question):
    """
    --- THIS IS THE UPDATED RAG FUNCTION ---
    It now returns (answer, context) or (error_message, None)
    """
    print(f"Answering RAG question: {user_question}")
    
    # 1. Retrieve relevant chunks from Cosdata
    retrieved_chunks = query_cosdata(user_question, top_k=5)
    
    # # --- DEBUG LINE ---
    # print("\n--- DEBUG: TOP 5 RETRIEVED CHUNKS ---")
    # for i, chunk in enumerate(retrieved_chunks):
    #     print(f"CHUNK {i+1}: {chunk[:150]}...") 
    # print("--- END DEBUG ---\n")
    # # ------------------------
    
    if not retrieved_chunks:
        # This now correctly returns two values (a string and None)
        return "I'm sorry, I couldn't find any relevant information in the document to answer that question.", None
    
    # 2. Combine chunks into a context string
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 3. Build a new prompt for the LLM
    # --- THIS PROMPT IS NOW FINE-TUNED (STRATEGY 2) ---
    prompt = f"""You are an expert legal Q&A assistant. Your task is to answer the user's question based *only* on the context snippets provided below.

    Follow these rules:
    1.  **Factual Questions (e.g., "What", "When", "How"):** Answer the question directly using facts from the context.
    2.  **Advice Questions (e.g., "Should I", "Can I"):** You MUST NOT provide legal advice. Instead, state what the document says factually and then add a disclaimer that you cannot provide advice.
    3.  **Corrections:** Correct obvious OCR errors (e.g., 'af' should be 'of').
    4.  **Limits:** If the answer is not in the context, say so clearly.

    **Example for Rule 2:**
    * **User Asks:** "Should I cancel my contract?"
    * **Correct Response:** "The document states that either party may terminate with 14 days' notice. However, I cannot provide legal advice on whether you *should* cancel your contract. Please consult a qualified lawyer."

    ----- CONTEXT SNIPPETS -----
    {context}
    ----- END OF CONTEXT SNIPPETS -----

    Based *only* on the context snippets and rules above, answer the following question:
    USER QUESTION: {user_question}
    ANSWER: """
    
    try:
        response = model.generate_content(prompt)
        # This returns 2 values
        return response.text
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return f"Sorry, an error occurred while generating the answer: {e}"