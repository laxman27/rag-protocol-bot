import os
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Ask the Protocol", layout="wide")
st.markdown("## 🏥 Healthcare Protocol Chatbot")
st.info("For training & simulation only.")

# ---------- SAFE LOADER ----------
def build_store():
    docs = []
    for file in os.listdir("pdfs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"pdfs/{file}")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedder.encode([c.page_content for c in chunks])

    st.session_state["chunks"] = chunks
    st.session_state["vectors"] = vectors
    st.session_state["embedder"] = embedder

if not all(k in st.session_state for k in ("chunks", "vectors", "embedder")):
    with st.spinner("Loading protocol PDFs..."):
        build_store()

# ---------- RETRIEVER ----------
def retrieve(query, k=4):
    if not all(k in st.session_state for k in ("chunks", "vectors", "embedder")):
        build_store()

    chunks = st.session_state["chunks"]
    vectors = st.session_state["vectors"]
    embedder = st.session_state["embedder"]

    q_vec = embedder.encode([query])
    sims = cosine_similarity(q_vec, vectors)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return "\n\n".join([chunks[i].page_content for i in top_idx])

prompt = ChatPromptTemplate.from_template("""
You are a medical training assistant.
Answer ONLY from the context.

Context:
{context}

Question: {question}
""")

llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# simple function instead of chain
def ask_rag(question):
    context = retrieve(question)
    messages = prompt.format_messages(context=context, question=question)
    return llm.invoke(messages)

# ---------- CHAT ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.markdown(f"**{role.upper()}:** {msg}")

q = st.text_input("Ask your protocol:")

if st.button("Send") and q:
    ans = ask_rag(q)
    st.session_state.chat.append(("you", q))
    st.session_state.chat.append(("bot", ans.content))
    st.rerun()

