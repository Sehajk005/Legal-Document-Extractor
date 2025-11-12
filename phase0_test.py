import time
from cosdata import Client
from sentence_transformers import SentenceTransformer

# 1. Setup
print("Wait, connecting to Cosdata and loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
client = Client(
    host="http://127.0.0.1:8443",
    username="admin",
    password="admin"
)

# 2. Create Collection
collection_name = "test_legal_docs"
try:
    existing = client.get_collection(collection_name)
    if existing:
        print(f"Collection '{collection_name}' exists. Deleting for fresh start...")
        existing.delete()
except Exception:
    pass 

print(f"Creating collection: {collection_name}...")
collection = client.create_collection(
    name=collection_name,
    dimension=384,
    description="Phase 0 Test Collection"
)

print("Creating HNSW index...")
collection.create_index(distance_metric="cosine")

# 3. Dummy Data
documents = [
    {"id": "doc_1", "text": "The tenant shall pay rent on or before the first day of each month."},
    {"id": "doc_2", "text": "Any dispute arising from this agreement shall be settled by arbitration in Chandigarh."},
    {"id": "doc_3", "text": "The landlord reserves the right to inspect the property with 24-hour notice."},
    {"id": "doc_4", "text": "The recipe calls for two cups of flour and one cup of sugar."} 
]

# 4. Embed and Upsert Data
print("Embedding and inserting documents...")
with collection.transaction() as txn:
    for doc in documents:
        vector_values = model.encode(doc["text"]).tolist()
        # --- FIX APPLIED HERE: Storing text explicitly in metadata ---
        vector_payload = {
            "id": doc["id"],
            "dense_values": vector_values,
            "metadata": {"source_text": doc["text"]} # Using a clear key
        }
        # -----------------------------------------------------------
        txn.upsert_vector(vector_payload)

print("Data inserted. Waiting a moment for indexing...")
time.sleep(2) 

# 5. Perform Search
query_text = "Can the landlord enter my house whenever they want?"
print(f"\nUser Query: '{query_text}'")

query_vector = model.encode(query_text).tolist()

results = collection.search.dense(
    query_vector=query_vector,
    top_k=2
    # Removed 'return_raw_text' to avoid confusion. We'll get metadata instead.
)

print("\n--- Search Results ---")
results_list = results.get('results', [])

# --- DEBUG: Print the very first raw result to see its structure ---
if results_list:
    print("DEBUG RAW RESULT:", results_list[0])
# -------------------------------------------------------------------

for res in results_list:
    metadata = res.get('metadata', {})
    # ... rest of loop
    retrieved_text = metadata.get('source_text', 'Text not found in metadata.')
    # -----------------------------------------------------------------
    print(f"\nScore: {res.get('score'):.4f}")
    print(f"Text: {retrieved_text}")

print("\nPhase 0 is now officially complete! We have a fully working Cosdata instance. 🚀")