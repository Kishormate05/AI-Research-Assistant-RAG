import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embedding_model

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

embeddings = get_embedding_model()

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = input("Ask a Question: ")

results = db.similarity_search(query, k=3)

context = "\n\n".join(
    [doc.page_content for doc in results]
)

prompt = f"""
Answer only from the provided context.

Context:
{context}

Question:
{query}
"""

response = model.generate_content(prompt)

print("\nAnswer:\n")
print(response.text)