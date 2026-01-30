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

# ---------- UI ----------
st.set_page_config(page_title="Ask the Protocol", layout="wide")
st.markdown("## 🏥 Healthcare Protocol Chatbot")
st.info("For training & simulation only.")

# ---------- LOAD & EMBED ----------
if "ready" not in st.session_state:
    with st.spinner("Loading protocol PDFs..."):
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
        st.session_state["ready"] = True

# ---------- RETRIEVER ----------
def retrieve(query, k=4):
    embedder = st.session_state["embedder"]
    vectors = st.session_state["vectors"]
    chunks = st.session_state["chunks"]

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

llm = ChatGroq(model="llama3-70b-8192")

rag_chain = (
    {"context": lambda q: retrieve(q), "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ---------- CHAT ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.markdown(f"**{role.upper()}:** {msg}")

q = st.text_input("Ask your protocol:")

if st.button("Send") and q:
    ans = rag_chain.invoke(q)
    st.session_state.chat.append(("you", q))
    st.session_state.chat.append(("bot", ans.content))
    st.rerun()
