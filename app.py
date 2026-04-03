import streamlit as st
from orchestrator import run
from rag.query import retrieve 
from memory import store_case,search memory 
from pdf_ingest import extract_text_from_pdf

St.set_page_config(page_title ="JARVIS Legal AI", layout="wide")
st.title(