import faiss
import numpy as np
from rag.embed import embed_text

index = faiss.read_index("rag/ucc.index")

with open("rag/chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.readlines()

def retrieve(query, top_k=3):
    q_emb = np.array([embed_text(query)]).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = [chunks[i] for i in I[0]]
    return "\n".join(results)
