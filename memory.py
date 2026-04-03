import faiss
import numpy as np
import os
from rag.embed import embed_text

MEMORY_INDEX = "memory.index"
MEMORY_TEXT = "memory.txt"

if os.path.exists(MEMORY_INDEX):
    index = faiss.read_index(MEMORY_INDEX)
else:
    index = faiss.IndexFlatL2(768)

memory_chunks = []

if os.path.exists(MEMORY_TEXT):
    with open(MEMORY_TEXT, "r", encoding="utf-8") as f:
        memory_chunks = f.readlines()

def chunk(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def store_case(text):
    global memory_chunks
    chunks = chunk(text)

    embeddings = [embed_text(c) for c in chunks]
    index.add(np.array(embeddings).astype("float32"))

    memory_chunks.extend(chunks)

    faiss.write_index(index, MEMORY_INDEX)

    with open(MEMORY_TEXT, "w", encoding="utf-8") as f:
        for c in memory_chunks:
            f.write(c.replace("\n", " ") + "\n")

def search_memory(query, top_k=3):
    if not memory_chunks:
        return "No memory yet."

    q_emb = np.array([embed_text(query)]).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = [memory_chunks[i] for i in I[0]]
    return "\n".join(results)