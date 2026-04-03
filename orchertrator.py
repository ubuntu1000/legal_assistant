import requests
from rag.query import retrieve
from memory import search_memory

OPENCLAW_URL = "http://localhost:5000/generate"

def run(user_input):
    rag_context = retrieve(user_input)
    memory_context = search_memory(user_input)

    prompt = f'''
You are a legal AI assistant.

Memory:
{memory_context}

Context:
{rag_context}

Question:
{user_input}
'''

    try:
        res = requests.post(
            OPENCLAW_URL,
            json={"prompt": prompt, "model": "phi3"}
        ).json()

        return res.get("response", "No response")
    except Exception as e:
        return f"Error: {e}"
