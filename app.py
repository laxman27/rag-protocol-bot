import os
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Ask the Protocol", layout="wide")
st.markdown("## 🏥 Healthcare Protocol Chatbot")
st.info("For training & simulation only.")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------- LOAD & EMBED ----------
def load_docs():
    texts = []
    for file in os.listdir("pdfs"):
        if file.endswith(".pdf"):
            reader = PdfReader(f"pdfs/{file}")
            for page in reader.pages:
                texts.append(page.extract_text() or "")
    return texts

if "texts" not in st.session_state:
    with st.spinner("Indexing protocol PDFs..."):
        texts = load_docs()
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = embedder.encode(texts)

        st.session_state["texts"] = texts
        st.session_state["vectors"] = vectors
        st.session_state["embedder"] = embedder

# ---------- RETRIEVER ----------
def retrieve(query, k=4):
    embedder = st.session_state["embedder"]
    vectors = st.session_state["vectors"]
    texts = st.session_state["texts"]

    q_vec = embedder.encode([query])
    sims = cosine_similarity(q_vec, vectors)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return "\n\n".join([texts[i] for i in top_idx])

# ---------- GROQ CHAT ----------
def ask_rag(question):
    context = retrieve(question)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a medical training assistant. Answer only from the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# ---------- CHAT UI ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.markdown(f"**{role.upper()}:** {msg}")

q = st.text_input("Ask your protocol:")

if st.button("Send") and q:
    ans = ask_rag(q)
    st.session_state.chat.append(("you", q))
    st.session_state.chat.append(("bot", ans))
    st.rerun()
