import os
import time
from cosdata import Client
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
COLLECTION_NAME = "legal_aid_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COSDATA_HOST = "http://127.0.0.1:8443"
# --------------

print(f"Loading embedding model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)
client = Client(host=COSDATA_HOST, username="admin", password="admin")

def get_or_create_collection():
    """Ensures the collection exists and has an index."""
    
    
    try:
        existing = client.get_collection(COLLECTION_NAME)
        if existing:
            return existing
    except Exception:
        pass 

    print(f"Creating new collection: {COLLECTION_NAME}")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        dimension=384,
        description="RAG storage for Legal Aid App"
    )
    collection.create_index(distance_metric="cosine")
    return collection

def smart_chunker(text, chunk_size=800, overlap=100):
    """
    A more robust chunker that splits by sentences and then words
    to respect a chunk size.
    """
    chunks = []
    start = 0

    # Iterate through the text, moving the start window
    while start < len(text):
        # Calculate the end of the chunk
        end = start + chunk_size
        
        # Take the slice (it's okay if it goes past the end)
        chunk = text[start:end]
        
        # Add the chunk if it's not just whitespace
        if chunk.strip():
            chunks.append(chunk)
            
        # Move the start pointer forward for the next chunk
        start += (chunk_size - overlap)
        
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def index_document(full_text, doc_name="uploaded_doc"):
    collection = get_or_create_collection()
    chunks = smart_chunker(full_text)
    
    print(f"Indexing {len(chunks)} chunks for {doc_name}...")
    txn = collection.create_transaction()
    try:
        for i, chunk in enumerate(chunks):
            vector_id = f"{doc_name}_chunk_{i}"
            embedding = model.encode(chunk).tolist()
            
            # We prefer using the standard 'text' field if possible, 
            # but our Two-Step fetch will grab whatever works.
            # Let's try storing it in BOTH text and document_id to be absolutely sure.
            payload = {
                "id": vector_id,
                "dense_values": embedding,
                "document_id": chunk, # Backup storage
                "text": chunk         # Primary storage
            }
            txn.upsert_vector(payload)
        
        print("Committing transaction...")
        txn.commit()
        print("Transaction committed.")
    except Exception as e:
        print(f"Error during indexing: {e}")
        txn.abort()
        raise e
            
    return len(chunks)

def query_cosdata(user_question, top_k=3):
    collection = get_or_create_collection()
    query_vec = model.encode(user_question).tolist()
    
    print(f"Searching for: '{user_question}'...")
    # STEP 1: Get the IDs of relevant documents
    search_results = collection.search.dense(
        query_vector=query_vec,
        top_k=top_k
    )
    
    final_results = []
    print("Fetching full documents...")
    for res in search_results.get('results', []):
        vec_id = res.get('id')
        if vec_id:
            # STEP 2: Explicitly fetch the full data for this ID
            try:
                full_doc = collection.vectors.get(vec_id)
                # Try to grab text from our primary or backup slots
                # Note: Depending on SDK, full_doc might be an object or dict.
                if isinstance(full_doc, dict):
                     text = full_doc.get('text') or full_doc.get('document_id')
                else:
                     # Assume it's an object with attributes
                     text = getattr(full_doc, 'text', None) or getattr(full_doc, 'document_id', None)

                if text:
                    final_results.append(text)
            except Exception as e:
                print(f"Failed to fetch document {vec_id}: {e}")
            
    return final_results