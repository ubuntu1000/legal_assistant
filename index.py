import faiss
import numpy as np
from rag.embed import embed_text

def chunk_text(text, size=300):
    return [text[i:i+size] for i in range(0, len(text), size)]

def build_index():
    with open("data/ucc_uttarakhand.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = [embed_text(c) for c in chunks]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, "rag/ucc.index")

    with open("rag/chunks.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.replace("\n", " ") + "\n")

if __name__ == "__main__":
    build_index()
