import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

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
# LOAD VECTOR DB ONCE
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


db = load_vector_db()

# -----------------------
# UI
# -----------------------

st.title("📚 AI Research Assistant")

st.markdown(
    "Ask questions from your uploaded Reinforcement Learning PDF."
)

question = st.text_input(
    "Ask a Question"
)

# -----------------------
# QUESTION ANSWERING
# -----------------------

if question:

    with st.spinner("Searching documents and generating answer..."):

        results = db.similarity_search(
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

        response = model.generate_content(prompt)

    # -----------------------
    # ANSWER
    # -----------------------

    st.subheader("📌 Answer")

    st.success(response.text)

    # -----------------------
    # SOURCES
    # -----------------------

    st.subheader("📄 Retrieved Sources")

    for i, doc in enumerate(results, start=1):

        with st.expander(f"Source {i}"):

            st.write(doc.page_content[:1000])