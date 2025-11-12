import time
import json
from cosdata import Client

# We don't even need real embeddings for this test, just dummy lists
DUMMY_VEC = [0.1] * 384 

client = Client(host="http://127.0.0.1:8443", username="admin", password="admin")
COLLECTION_NAME = "debug_probe"

# 1. Clean Slate
try:
    client.get_collection(COLLECTION_NAME).delete()
    print("Deleted old debug collection.")
except:
    pass

# 2. Create Collection
print("Creating debug collection...")
collection = client.create_collection(name=COLLECTION_NAME, dimension=384)

# 3. The PROBE: Try ONLY valid formats now
print("Probing valid insertion formats...")
try:
    with collection.transaction() as txn:
        # Attempt 1: Top-level 'text' field
        txn.upsert_vector({
            "id": "probe_text", 
            "dense_values": DUMMY_VEC, 
            "text": "Success: Data in 'text' field"
        })
        # Attempt 2: Top-level 'metadata' dictionary
        txn.upsert_vector({
            "id": "probe_metadata", 
            "dense_values": DUMMY_VEC, 
            "metadata": {"result": "Success: Data in 'metadata' field"}
        })
        # Attempt 3: Using 'document_id'
        txn.upsert_vector({
            "id": "probe_doc_id", 
            "dense_values": DUMMY_VEC, 
            "document_id": "Success_Data_in_document_id_field" 
        })

    print("Commit and wait...")
    time.sleep(2)

    # 4. Direct Fetch
    print("\n--- PROBE RESULTS ---")
    for probe_id in ["probe_text", "probe_metadata", "probe_doc_id"]:
        try:
            vector = collection.vectors.get(probe_id)
            print(f"\nID: {probe_id}")
            # Use a safe way to print the vector object/dict
            val = vector if isinstance(vector, dict) else vars(vector)
            print(json.dumps(val, indent=2, default=str))
        except Exception as e:
            print(f"\nID: {probe_id} - FAILED TO FETCH: {e}")

except Exception as e:
    print(f"\nCRITICAL FAILURE DURING TRANSACTION: {e}")

print("\nProbe complete.")