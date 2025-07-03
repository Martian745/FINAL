# --- 📁 Deep Scraper for MOSDAC + Integration into Streamlit App ---

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader

# Configure scraper settings
BASE_URL = "https://www.mosdac.gov.in"
SCRAPE_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "mosdac_scraped")
os.makedirs(SCRAPE_OUTPUT_DIR, exist_ok=True)

visited_links = set()
documents = []

# --- Util: Download and extract PDF content ---
def download_and_parse_pdf(pdf_url):
    try:
        filename = os.path.basename(urlparse(pdf_url).path)
        local_path = os.path.join(SCRAPE_OUTPUT_DIR, filename)
        response = requests.get(pdf_url)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        pdf_docs = PyPDFLoader(local_path).load()
        for doc in pdf_docs:
            doc.metadata['source'] = pdf_url
        return pdf_docs
    except Exception as e:
        print(f"PDF Error at {pdf_url}: {e}")
        return []

# --- Core Recursive Crawler ---
def scrape_mosdac(url, progress_callback=None):
    if url in visited_links or not url.startswith(BASE_URL):
        return

    try:
        print(f"🔍 Scraping: {url}")
        visited_links.add(url)
        if progress_callback:
            progress_callback(f"🔍 Scraping: {url}")

        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Save raw HTML text as a Document
        text = soup.get_text(separator="\n", strip=True)
        documents.append(Document(page_content=text, metadata={"source": url}))

        # Extract and follow links
        for link_tag in soup.find_all("a", href=True):
            href = urljoin(url, link_tag['href'])
            if href.endswith(".pdf"):
                documents.extend(download_and_parse_pdf(href))
            elif href.startswith(BASE_URL):
                scrape_mosdac(href, progress_callback)

        time.sleep(1)  # be polite

    except Exception as e:
        print(f"❌ Failed on {url}: {e}")
        if progress_callback:
            progress_callback(f"❌ Failed on {url}: {e}")

# --- Integration Trigger Function ---
@st.cache_data(show_spinner=False)
def load_mosdac_website():
    status = st.empty()
    with st.spinner("🕸️ Deep crawling MOSDAC website, please wait..."):
        status.text("🚀 Starting deep scrape of MOSDAC website...")
        documents.clear()
        visited_links.clear()
        scrape_mosdac(BASE_URL, progress_callback=status.text)
        status.text(f"✅ Scraped {len(documents)} documents from MOSDAC.")
    return documents

# --- Streamlit UI for Trigger ---
st.markdown("### 🌐 Or Deep Scrape the MOSDAC Website")

if st.button("🕷️ Start Deep Scrape of MOSDAC"):
    docs = load_mosdac_website()

    if docs:
        st.success(f"✅ Scraped {len(docs)} documents from MOSDAC.")
        st.markdown("#### 🔍 Sample Scraped Content:")
        for i, doc in enumerate(docs[:3]):
            st.markdown(f"**Document {i+1} — Source:** {doc.metadata.get('source')}")
            st.text(doc.page_content[:500] + "...")

import os
import streamlit as st
import tempfile
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
import re
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load Groq API Key
groq_api_key = st.secrets["groq"]["api_key"] if "groq" in st.secrets else os.getenv("GROQ_API_KEY")

# Similarity Function
def calculate_similarity(text1, text2):
    import re
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    text1_clean = normalize(text1)
    text2_clean = normalize(text2)

    embeddings = model.encode([text1_clean, text2_clean])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Load content
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name

elif url_input:
    loader = WebBaseLoader(url_input)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url_input

if docs:
    st.success("✅ Content Loaded!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"device": "cpu"}  # Force device assignment to avoid meta tensor errors
)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    except Exception as e:
        st.error(f"LLM setup failed: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📊 Knowledge Graph", "🧪 Evaluate Bot Accuracy"])

    with tab1:
        st.subheader("Ask your question about the document")
        user_input = st.text_input("🔍 Enter your question:")
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain({"query": user_input})
                    st.markdown(f"🤖 **Answer:** {result['result']}")
                    for src in result["source_documents"]:
                        st.caption(f"📎 Source: {src.metadata.get('source', 'Unknown')}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with tab2:
        st.subheader("Auto-generated Knowledge Graph")
        if st.button("🔄 Generate Graph"):
            nlp = spacy.load("en_core_web_sm")
            full_text = "\n".join([doc.page_content for doc in docs])
            doc_nlp = nlp(full_text)
            G = nx.DiGraph()
            for sent in doc_nlp.sents:
                subject = obj = verb = ""
                for token in sent:
                    if token.dep_ in ("nsubj", "nsubjpass"):
                        subject = token.text
                        verb = token.head.lemma_
                    if token.dep_ == "dobj":
                        obj = token.text
                if subject and obj:
                    G.add_edge(subject, obj, label=verb)

            if not G.nodes:
                st.warning("📉 No clear relationships extracted.")
            else:
                plt.figure(figsize=(14, 8))
                pos = nx.spring_layout(G, k=0.5)
                nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=9)
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                st.pyplot(plt)

    with tab3:
        st.subheader("🧪 Evaluate Bot Against Answer Key PDF")
        if eval_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(eval_file.read())
                eval_path = tmp.name
            qa_text = "\n".join([p.page_content for p in PyPDFLoader(eval_path).load()])

            if auto_clean:
    # More robust parser for ===QUESTION=== and ===ANSWER=== format
    qa_pairs = []
    qa_blocks = re.split(r"===QUESTION===", qa_text)
    for block in qa_blocks:
        parts = block.strip().split("===ANSWER===")
        if len(parts) == 2:
            question, answer = parts
            question = question.strip().replace("
", " ")
            answer = answer.strip().replace("
", " ")
            qa_pairs.append((question, answer))
            else:
                qa_lines = [line.strip() for line in qa_text.split("\n") if line.strip()]
                qa_pairs = [(qa_lines[i], qa_lines[i + 1]) for i in range(0, len(qa_lines) - 1, 2)]

            with st.spinner("🧠 Evaluating chatbot performance..."):
                report = []
                total_questions = len(qa_pairs)
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, (question, expected_answer) in enumerate(qa_pairs, start=1):
                    status_text.text(f"🔄 Evaluating Question {idx} of {total_questions}...")
                    try:
                        response = qa_chain({"query": question})
                        bot_answer = response["result"]
                        score = calculate_similarity(expected_answer, bot_answer)

                        comment = (
                            "✅ Excellent match" if score >= 0.85 else
                            "🟡 Partial match" if score >= 0.65 else
                            "⚠️ Weak match" if score >= 0.4 else
                            "❌ Poor match"
                        )

                        report.append({
                            "Question": question,
                            "Reference Answer": expected_answer,
                            "Bot Answer": bot_answer,
                            "Similarity Score": round(score, 3),
                            "Comment": comment
                        })
                    except Exception as e:
                        report.append({
                            "Question": question,
                            "Reference Answer": expected_answer,
                            "Bot Answer": "ERROR",
                            "Similarity Score": 0.0,
                            "Comment": f"❌ Error: {e}"
                        })
                        st.error(f"⚠️ Error on Question {idx}: {e}")

                    progress_bar.progress(idx / total_questions)
                    time.sleep(0.1)

                st.success("🎉 Evaluation complete!")
                df = pd.DataFrame(report)
                avg_score = df["Similarity Score"].mean()
                accuracy = (df["Similarity Score"] >= 0.75).mean() * 100

                st.markdown("## 📋 Evaluation Summary")
                st.markdown(f"- **Average Similarity Score:** `{avg_score:.2f}`")
                st.markdown(f"- **Answers Above 75% Similarity:** `{accuracy:.1f}%`")
                st.markdown(f"- **Total Questions Evaluated:** `{total_questions}`")

                st.subheader("📊 Detailed Question-wise Evaluation")
                st.dataframe(df)

                st.subheader("🔍 Similarity Score Visualization")
                st.bar_chart(df["Similarity Score"])

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Evaluation Results as CSV",
                    data=csv,
                    file_name="bot_evaluation_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("""📄 Upload a Q&A reference PDF using one of the following formats:
1. Clean alternating lines (Q line, then A line)
2. Compact format with Q:/A:
3. Block format using ===QUESTION=== and ===ANSWER=== markers for multi-line content.
""")

