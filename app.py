import sys
import os

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import process_pdf_for_text
from src.information_extraction.extractor import extract_entities_with_llm
from src.information_extraction.extractor import answer_user_questions
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
    
st.title("AI Powered Legal Aid For Common Citizens")
upload_file = st.file_uploader("Upload a PDF", type=['pdf'])
if upload_file is not None:
    if st.button("Analyze Document", type="primary"):
        with st.spinner("Processing PDF... This may take a few minutes..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tem_file: # create a temporary file
                tem_file.write(upload_file.getvalue()) # write the uploaded file to the temporary file
                tem_file_path = tem_file.name # get the temporary file path
    
            text = process_pdf_for_text(tem_file_path)
    
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
    st.subheader("------Analysis Report------\n")
    
    # Add this line back for debugging
    #st.json(st.session_state.analysis_data)
    # --- USAGE ---
    report_data = st.session_state.analysis_data
    
    # Define lists of possible keys for each piece of data
    entity_key_aliases = ['extracted_entities', 'entities', 'entity_extraction']
    clause_key_aliases = ['extracted_clauses', 'clauses', 'clause_analysis']
    
    # Use the helper function to find the data, regardless of which key the LLM used
    entities_data = find_data(report_data, entity_key_aliases)
    clauses_data = find_data(report_data, clause_key_aliases)
    # --- END OF USAGE ---

    # Now we display the data found by our robust function
    st.markdown(f"**Individual Names:** {find_data(entities_data, ['individual_names', 'individuals'], [])}")
    st.markdown(f"**Dates:** {find_data(entities_data, ['dates'], [])}")
    st.markdown(f"**Addresses:** {find_data(entities_data, ['addresses_locations', 'addresses_or_locations', 'addresses'], [])}")
    st.markdown(f"**Phone Numbers:** {find_data(entities_data, ['phone_numbers'], [])}")
    st.markdown(f"**Emails:** {find_data(entities_data, ['emails'], [])}")
    st.markdown(f"**Company Names:** {find_data(entities_data, ['company_names', 'companies'], [])}")
    st.markdown(f"**Organization Names:** {find_data(entities_data, ['organization_names', 'organizations'], [])}")
    
    st.subheader(f"Found {len(clauses_data)} clauses:")

    for clause in clauses_data:
        with st.expander(f"**{clause.get('clause_title', 'Untitled Clause')}**"):
            st.markdown(f"""
            - **Clause Type:** *{clause.get('clause_type', 'N/A')}*
            - **Summary:** *{clause.get('summary_in_plain_english', 'N/A')}*
            - **Potential Risks:** *{clause.get('potential_risks', 'N/A')}*
            """)
            st.markdown("**Full Clause Text:**")
            st.write(clause.get('clause_text', 'N/A'))
    
    st.header("Ask questions regarding any queries from the document:")

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
                response = answer_user_questions(prompt)
                # ----------------------------------

                # Add assistant response to history
                st.session_state.message_history.append({"role": "assistant", "content": response})
                # This will be displayed in the next rerun
                st.rerun()