import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Streamlit UI setup
st.set_page_config(page_title="MOSDAC AI Bot (Offline)", layout="wide")
st.title("🛰️ MOSDAC AI Help Bot — Offline with Ollama")

# File or URL input
uploaded_file = st.file_uploader("📄 Upload PDF or DOCX file", type=["pdf", "docx"])
url_input = st.text_input("🌐 Or paste a web URL")

docs = []

# Load documents
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(tmp_path)
        docs = loader.load()
elif url_input:
    loader = WebBaseLoader(url_input)
    docs = loader.load()

# If documents loaded
if docs:
    st.success("✅ Content loaded!")

    # Text chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Local embeddings via Ollama
    #embeddings = OllamaEmbeddings()
    embeddings = OllamaEmbeddings(model="mistral")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # Local LLM
    llm = Ollama(model="mistral")  # or "llama3", "gemma", etc.
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # UI Tabs
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Knowledge Graph"])

    with tab1:
        st.subheader("Ask your question about the document")
        user_input = st.text_input("🔍 Enter your question:")
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain.run(user_input)
                    st.markdown(f"**🤖 Answer:** {response}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with tab2:
        st.subheader("Auto Knowledge Graph")
        if st.button("🔄 Generate Graph"):
            nlp = spacy.load("en_core_web_sm")
            full_text = "\n".join([doc.page_content for doc in docs])
            doc_spacy = nlp(full_text)

            G = nx.Graph()
            for sent in doc_spacy.sents:
                subj = ""
                obj = ""
                for token in sent:
                    if "subj" in token.dep_:
                        subj = token.text
                    if "obj" in token.dep_:
                        obj = token.text
                if subj and obj:
                    G.add_edge(subj, obj)

            if not G.nodes:
                st.warning("📉 No relationships found.")
            else:
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G, k=0.5)
                nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray',
                        node_size=2000, font_size=10)
                st.pyplot(plt)
else:
    st.info("📂 Upload a file or enter a URL to begin.")
