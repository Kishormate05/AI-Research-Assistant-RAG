import os
import tempfile
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.embeddings import get_embedding_model

# -----------------------
# CONFIG
# -----------------------

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# -----------------------
# LOAD DEFAULT VECTOR DB
# -----------------------

@st.cache_resource
def load_vector_db():

    embeddings = get_embedding_model()

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


default_db = load_vector_db()

# -----------------------
# SESSION STATE
# -----------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# TITLE
# -----------------------

st.title("📚 AI Research Assistant")

st.markdown(
    "Upload one or more PDFs and ask questions using RAG + Gemini."
)

# -----------------------
# MULTI PDF UPLOAD
# -----------------------

uploaded_files = st.file_uploader(
    "📤 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

uploaded_db = None
total_chunks = 0

if uploaded_files:

    all_docs = []

    with st.spinner("Processing PDFs..."):

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.getbuffer()
                )

                pdf_path = tmp_file.name

            loader = PyPDFLoader(
                pdf_path
            )

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(
            all_docs
        )

        total_chunks = len(chunks)

        embeddings = get_embedding_model()

        uploaded_db = FAISS.from_documents(
            chunks,
            embeddings
        )

    st.success(
        f"✅ Processed {len(uploaded_files)} PDF(s) | Total Chunks: {total_chunks}"
    )

# -----------------------
# SIDEBAR DASHBOARD
# -----------------------

with st.sidebar:

    st.title("📊 Dashboard")

    st.markdown("---")

    st.metric(
        "📚 PDFs Loaded",
        len(uploaded_files) if uploaded_files else 0
    )

    st.metric(
        "📄 Total Chunks",
        total_chunks
    )

    st.markdown("---")

    if uploaded_files:

        st.subheader("Uploaded PDFs")

        for pdf in uploaded_files:

            st.write(f"📄 {pdf.name}")

    st.markdown("---")

    st.subheader("System")

    st.write("🤖 Gemini 2.5 Flash")
    st.write("🔍 FAISS Vector Search")
    st.write("🧠 MiniLM-L6-v2 Embeddings")

# -----------------------
# QUESTION
# -----------------------

question = st.text_input(
    "Ask a Question"
)

# -----------------------
# ANSWERING
# -----------------------

if question:

    with st.spinner(
        "Searching documents and generating answer..."
    ):

        if uploaded_db:

            results = uploaded_db.similarity_search(
                question,
                k=5
            )

        else:

            results = default_db.similarity_search(
                question,
                k=5
            )

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        prompt = f"""
Answer only from the provided context.

Context:
{context}

Question:
{question}
"""

        response = model.generate_content(
            prompt
        )

        answer = response.text

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

    # -----------------------
    # ANSWER
    # -----------------------

    st.subheader("📌 Answer")

    st.success(answer)

    # -----------------------
    # SOURCES
    # -----------------------

    st.subheader("📄 Retrieved Sources")

    for i, doc in enumerate(results, start=1):

        source_name = doc.metadata.get(
            "source",
            "Unknown PDF"
        )

        with st.expander(
            f"Source {i} - {source_name}"
        ):

            st.write(
                doc.page_content[:1500]
            )

# -----------------------
# CHAT HISTORY
# -----------------------

if st.session_state.chat_history:

    st.markdown("---")

    st.subheader("💬 Chat History")

    for idx, item in enumerate(
        reversed(st.session_state.chat_history),
        start=1
    ):

        with st.expander(
            f"Q{idx}: {item['question']}"
        ):

            st.write("**Answer:**")
            st.write(item["answer"])