import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# ===============================
# 🔹 STEP 1: Load JSON
# ===============================
with open("call_titles.json", "r") as f:
    data = json.load(f)

titles = list(set(item["call_title"] for item in data))

print(f"✅ Loaded {len(titles)} titles")

# ===============================
# 🔹 STEP 2: Load Embedding Model
# ===============================
print("⏳ Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ===============================
# 🔹 STEP 3: Generate Embeddings
# ===============================
print("⏳ Generating embeddings...")
embeddings = model.encode(titles)

# Convert to numpy float32
embeddings = np.array(embeddings).astype("float32")

# ===============================
# 🔹 STEP 4: Create FAISS Index
# ===============================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add data
index.add(embeddings)

print("✅ FAISS index created")

# ===============================
# 🔹 STEP 5: Save Index + Titles
# ===============================
faiss.write_index(index, "faiss_index.index")

with open("titles.pkl", "wb") as f:
    pickle.dump(titles, f)

print("✅ Data saved successfully!")

# ===============================
# 🔹 DONE
# ===============================
print("\n🎉 Everything completed successfully!")