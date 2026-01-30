import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from deep_translator import GoogleTranslator

# ---------- UI ----------
st.set_page_config(page_title="Ask the Protocol", layout="wide")
st.markdown("""
<style>
body { background-color:#0e1117; color:white; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🏥 Healthcare Protocol Chatbot")
st.info("For training & simulation only. Follow official hospital protocols.")

# ---------- LOAD RAG ----------
if "db" not in st.session_state:
    with st.spinner("Loading protocol PDFs..."):
        docs = []
        for file in os.listdir("pdfs"):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(f"pdfs/{file}")
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="llama3")
        db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
        st.session_state.db = db

retriever = st.session_state.db.as_retriever()

prompt = ChatPromptTemplate.from_template("""
You are a medical training assistant.
Answer ONLY from the context.

Context:
{context}

Question: {question}
""")

llm = Ollama(model="llama3")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ---------- CHAT ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.markdown(f"**{role.upper()}:** {msg}")

question = st.text_input("Ask your protocol:")

if st.button("Send") and question:
    st.session_state.chat.append(("you", question))

    user_lang = GoogleTranslator().detect(question)
    if user_lang != "en":
        question = GoogleTranslator(source='auto', target='en').translate(question)

    answer = rag_chain.invoke(question)

    if user_lang == "te":
        answer = GoogleTranslator(source='en', target='te').translate(answer)

    st.session_state.chat.append(("bot", str(answer)))
    st.rerun()
