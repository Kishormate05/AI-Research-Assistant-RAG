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
# UI
# -----------------------

st.title("📚 AI Research Assistant")

st.markdown(
    "Upload any PDF and ask questions using RAG + Gemini."
)

# -----------------------
# PDF UPLOAD
# -----------------------

uploaded_file = st.file_uploader(
    "📤 Upload PDF",
    type=["pdf"]
)

uploaded_db = None

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.getbuffer()
        )

        pdf_path = tmp_file.name

    with st.spinner(
        "Processing PDF..."
    ):

        loader = PyPDFLoader(
            pdf_path
        )

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(
            docs
        )

        embeddings = get_embedding_model()

        uploaded_db = FAISS.from_documents(
            chunks,
            embeddings
        )

    st.success(
        f"PDF processed successfully! Total Chunks: {len(chunks)}"
    )

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
                k=3
            )

        else:

            results = default_db.similarity_search(
                question,
                k=3
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

    # -----------------------
    # ANSWER
    # -----------------------

    st.subheader(
        "📌 Answer"
    )

    st.success(
        response.text
    )

    # -----------------------
    # SOURCES
    # -----------------------

    st.subheader(
        "📄 Retrieved Sources"
    )

    for i, doc in enumerate(
        results,
        start=1
    ):

        with st.expander(
            f"Source {i}"
        ):

            st.write(
                doc.page_content[:1500]
            )