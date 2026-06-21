from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embeddings import get_embedding_model

from langchain_community.vectorstores import FAISS

docs = load_pdf("data/rl_notes.pdf")

chunks = split_documents(docs)

embeddings = get_embedding_model()

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("vectorstore")

print("Vector Database Created Successfully!")
print("Total Chunks:", len(chunks))