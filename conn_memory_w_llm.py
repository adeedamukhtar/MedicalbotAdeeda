import os, sys, requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
#load_dotenv()
load_dotenv()
key = os.getenv("GROQ_API_KEY")
if not key or not key.startswith("gsk_"):
    sys.exit("GROQ_API_KEY missing or not a Groq key (should start with 'gsk_').")

# Quick auth sanity check
r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"}, timeout=10
)
if r.status_code != 200:
    sys.exit(f"Groq auth check failed: {r.status_code} {r.text}")
print("Groq auth OK.")


# Check if GROQ_API_KEY exists
load_dotenv(dotenv_path=".env")   # explicitly load
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not loaded. Check .env placement/format.")
print("✅ Loaded key:", GROQ_API_KEY[:6], "...")

# Initialize Groq LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful medical assistant.
Use the provided context to answer the question accurately and concisely.
If the context does not fully answer the question, say:
"Based on the available information, here is what I found..."

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])


# Load FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Run chatbot
try:
    user_query = input("Write Query Here: ")

    if not user_query.strip():
        raise ValueError("Query cannot be empty!")

    response = qa_chain.invoke({"query": user_query})

    print("\nRESULT:")
    print(response["result"])

    print("\nSOURCE DOCUMENTS:")
    for doc in response["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"))

except Exception as e:
    print(f"\nError occurred: {e}")
