import sys
import os
import uuid

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import process_pdf_for_text
from src.information_extraction.extractor import extract_entities_with_llm
from src.information_extraction.extractor import answer_user_questions
from src.cosdata_store import nuke_and_recreate_collection
import tempfile
import json
import re
import streamlit as st 
import pandas as pd
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

# HELPER FUNCTION FOR ROBUST DATA EXTRACTION
def find_data(data_dict, key_aliases, default_value=None):
    """
    Searches a dictionary for a list of possible keys and returns the value of the first key found.
    
    Args:
        data_dict (dict): The dictionary to search in.
        key_aliases (list): A list of possible keys to look for (e.g., ['entities', 'extracted_entities']).
        default_value: The value to return if no keys are found.
        
    Returns:
        The value found or the default value.
    """
    if default_value is None:
        # Infer the default value type from the first alias (dict or list)
        if any('clauses' in key or 'analysis' in key for key in key_aliases):
             default_value = []
        else:
            default_value = {}
            
    for key in key_aliases:
        if key in data_dict:
            return data_dict[key]
    return default_value

# Establish a connection to Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)
def log_feedback(question, answer, rating, comment=""):
    
    existing_feedback = conn.read(worksheet="Feedback", usecols=list(range(5)), ttl=0)
    
    # Create a new row of data as a DataFrame
    new_feedback = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": answer,
                "rating": rating,
                "comment": comment,
            }
        ]
    )
    ## Append the new feedback to the existing data
    updated_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
    
    # Update the worksheet with the new combined data
    conn.update(worksheet="Feedback", data=updated_feedback)
    

if 'message_history' not in st.session_state:
    st.session_state.message_history = []
    
# --- SET UP USER SESSION ---
if 'session_id' not in st.session_state:
    # Generate a clean, random ID for this user session
    # We use hex to avoid special characters that might break DB naming rules
    st.session_state.session_id = f"user_{uuid.uuid4().hex[:8]}"

# Define the user's unique collection name
user_collection = f"legal_aid_{st.session_state.session_id}"
# -------------------------
    
st.title("AI Powered Legal Aid For Common Citizens")
upload_file = st.file_uploader("Upload a PDF", type=['pdf'])
if upload_file is not None:
    if st.button("Analyze Document", type="primary"):
        with st.spinner("Processing PDF... This may take a few minutes..."):
            
            # 1. Save the file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tem_file: 
                tem_file.write(upload_file.getvalue()) 
                tem_file_path = tem_file.name 

            # 2. Capture the filename (This is our Filter Key)
            active_doc_name = os.path.basename(tem_file_path)
            st.session_state.active_doc_name = active_doc_name
            # Pass the SESSION ID (not the collection name)
            text = process_pdf_for_text(tem_file_path, st.session_state.session_id)

            # 3. Process (Pass a dummy collection name, we use Global now)
            text = process_pdf_for_text(tem_file_path, "global")
    
            llm_output = extract_entities_with_llm(text)
            data = {}
            json_data = ""
            try:
                # First, try to load the entire string as JSON. This handles the new clean output.
                data = json.loads(llm_output)
                json_data = llm_output
            except json.JSONDecodeError:
                # If that fails, fall back to searching for a markdown block.
                match = re.search(r'```(json)?\s*(\{.*\}|\[.*\])\s*```', llm_output, re.DOTALL)
                if match:
                    json_data = match.group(2)
                    try:
                        data = json.loads(json_data)
                    except json.JSONDecodeError as e:
                        st.error(f"Found a JSON block, but it's invalid: {e}")
                        st.code(llm_output, language='text')
                else:
                    st.error("Could not parse the LLM's response as JSON.")
                    st.code(llm_output, language='text')

            if data:
                # Store the data in the Streamlit session
                st.session_state.document_text = text
                st.session_state.llm_output = json_data
                st.session_state.analysis_data = data
                st.session_state.analysis_complete = True
                
if st.session_state.get('analysis_complete'):
    st.divider()
    st.subheader("📝 Document Analysis Report")
    
    report_data = st.session_state.analysis_data
    entities_data = find_data(report_data, ['extracted_entities', 'entities', 'entity_extraction'])
    clauses_data = find_data(report_data, ['extracted_clauses', 'clauses', 'clause_analysis'])

    # --- HELPER TO REMOVE JSON BRACKETS ---
    def clean_text(val):
        if isinstance(val, list):
            return ", ".join(val)
        return val if val else "N/A"

    # --- DASHBOARD UI FOR ENTITIES ---
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🏢 **Parties & Organizations**")
        comps = find_data(entities_data, ['company_names', 'companies'], [])
        orgs = find_data(entities_data, ['organization_names', 'organizations'], [])
        inds = find_data(entities_data, ['individual_names', 'individuals'], [])
        
        st.markdown(f"**Companies:** {clean_text(comps)}")
        st.markdown(f"**Organizations:** {clean_text(orgs)}")
        st.markdown(f"**Individuals:** {clean_text(inds)}")

    with col2:
        st.warning("📍 **Dates & Locations**")
        addrs = find_data(entities_data, ['addresses_locations', 'addresses_or_locations', 'addresses'], [])
        dates = find_data(entities_data, ['dates'], [])
        
        st.markdown(f"**Addresses:** {clean_text(addrs)}")
        st.markdown(f"**Key Dates:** {clean_text(dates)}")

    # Contact info in an expander to save space
    with st.expander("📞 View Contact Details"):
        emails = find_data(entities_data, ['emails'], [])
        phones = find_data(entities_data, ['phone_numbers'], [])
        st.markdown(f"**Emails:** {clean_text(emails)}")
        st.markdown(f"**Phone Numbers:** {clean_text(phones)}")
    
    st.divider()

    # --- CLAUSES WITH RED RISKS ---
    st.subheader(f"📜 Identified Clauses ({len(clauses_data)})")

    for clause in clauses_data:
        title = clause.get('clause_title', 'Untitled Clause')
        risks = clause.get('potential_risks', 'N/A')
        
        with st.expander(f"**{title}**"):
            # Use columns inside the expander: Summary Left, Risk Right
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.markdown(f"**Type:** `{clause.get('clause_type', 'N/A')}`")
                st.markdown(f"**Summary:** {clause.get('summary_in_plain_english', 'N/A')}")
            
            with c2:
                st.markdown("🚨 **Potential Risks:**")
                # THIS MAKES IT RED
                st.markdown(f":red[{risks}]")

            st.markdown("---")
            st.caption("**Full Clause Text:**")
            st.text(clause.get('clause_text', 'N/A'))
    
    st.divider()
    st.header("💬 AI Legal Assistant")
    st.caption("Ask questions about the document. The AI analyzes specific clauses to answer.")

    # 1. Display the entire chat history on every rerun
    for index, message in enumerate(st.session_state.message_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Add feedback buttons ONLY to assistant messages
            if message["role"] == "assistant":
                # Find the user question that corresponds to this answer
                question_for_feedback = ""
                if index > 0 and st.session_state.message_history[index - 1]["role"] == "user":
                    question_for_feedback = st.session_state.message_history[index - 1]["content"]

                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"good_{index}"):
                        st.session_state[f"show_comment_for_{index}"] = True
                        st.session_state[f"rating_for_{index}"] = "good"

                with col2:
                    if st.button("👎", key=f"bad_{index}"):
                        st.session_state[f"show_comment_for_{index}"] = True
                        st.session_state[f"rating_for_{index}"] = "bad"

                if st.session_state.get(f"show_comment_for_{index}"):
                    comment = st.text_area(
                        "Please provide more details:", 
                        key=f"comment_{index}"
                    )
                    if st.button("Submit Feedback", key=f"submit_{index}"):
                        try:
                            rating = st.session_state.get(f"rating_for_{index}", "bad")
                            log_feedback(question=question_for_feedback, answer=message["content"], rating=rating, comment=comment)
                            st.toast("Thanks for your detailed feedback!")
                            st.session_state[f"show_comment_for_{index}"] = False
                            st.rerun()
                        except Exception as e:
                            st.error("An error occurred while logging feedback:")
                            st.exception(e)

    # 2. Get new user input
    if prompt := st.chat_input("Ask a question"):
        # Add user message to history
        st.session_state.message_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # --- THIS IS THE MODIFIED LOGIC ---
                # We no longer build 'full_context'.
                # We just call the new RAG-powered function.
                doc_name = st.session_state.get('active_doc_name', '')
                response = answer_user_questions(prompt, st.session_state.session_id, doc_name)
                # ----------------------------------

                # Add assistant response to history
                st.session_state.message_history.append({"role": "assistant", "content": response})
                # This will be displayed in the next rerun
                st.rerun()