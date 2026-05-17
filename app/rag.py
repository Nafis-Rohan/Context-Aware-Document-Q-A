import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import tempfile

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ─── STEP 1: Process & Store PDF ──────────────────────────────────────────────

def process_pdf(file_bytes: bytes, filename: str) -> int:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Extract text from PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Each chunk = 500 characters
        chunk_overlap=50      # 50 characters overlap between chunks (so context isn't lost)
    )
    chunks = splitter.split_documents(documents)

    # Convert chunks to vectors and store in ChromaDB
    embeddings = OllamaEmbeddings(
        model="deepseek-r1:1.5b",  # Small model for embeddings (faster)
        base_url=OLLAMA_BASE_URL
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=filename.replace(".pdf", "")
    )

    os.unlink(tmp_path)  # Delete temp file
    return len(chunks)

# ─── STEP 2: Answer Questions ──────────────────────────────────────────────────

def answer_question(question: str, filename: str, provider: str = None) -> dict:
    provider = provider or LLM_PROVIDER

    # Load the stored vectors from ChromaDB
    embeddings = OllamaEmbeddings(
        model="deepseek-r1:1.5b",
        base_url=OLLAMA_BASE_URL
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=filename.replace(".pdf", "")
    )

    # Find the most relevant chunks for the question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top 3 most relevant chunks
    relevant_docs = retriever.get_relevant_documents(question)
    sources = [doc.page_content[:100] for doc in relevant_docs]  # First 100 chars as preview

    # Build context from relevant chunks
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
"""

    # Send to chosen LLM
    if provider == "ollama":
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        answer = llm.invoke(prompt)

    elif provider == "gemini":
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        answer = response.text

    else:
        answer = "Invalid provider. Use 'ollama' or 'gemini'."

    return {
        "answer": answer,
        "provider_used": provider,
        "sources": sources
    }