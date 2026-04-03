import streamlit as st
from orchestrator import run
from rag.query import retrieve
from memory import store_case, search_memory
from pdf_ingest import extract_text_from_pdf

st.set_page_config(page_title="JARVIS Legal AI", layout="wide")

st.title("⚖️ JARVIS - Legal AI Assistant")

mode = st.sidebar.selectbox("Mode", ["Chat", "Upload Case", "Memory Search"])

if mode == "Chat":
    query = st.text_input("Ask something:")
    if query:
        context = retrieve(query)
        memory = search_memory(query)
        full_context = f"Memory:\n{memory}\n\nContext:\n{context}"
        response = run(query + "\n" + full_context)
        st.markdown("### 🤖 Answer")
        st.write(response)

elif mode == "Upload Case":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        store_case(text)
        st.success("✅ Case added to memory + RAG system")

elif mode == "Memory Search":
    query = st.text_input("Search past cases")
    if query:
        results = search_memory(query)
        st.write(results)
