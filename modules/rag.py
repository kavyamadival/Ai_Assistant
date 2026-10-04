import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
VECTOR_DIR = Path("vectorstore/faiss_index")

def _get_llm():
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in .env")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0
    )

def _get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore():
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    documents = []
    for path in DATA_DIR.glob("*.txt"):
        if path.stat().st_size == 0:
            continue
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = path.name
        documents.extend(docs)

    if not documents:
        raise ValueError("No filled .txt files were found in the data folder.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_DIR))

    return f"Knowledge base created with {len(chunks)} chunks from {len(documents)} documents."

def _load_vectorstore():
    from langchain_community.vectorstores import FAISS
    if not VECTOR_DIR.exists():
        raise FileNotFoundError("Knowledge base not found. Click 'Build / Rebuild Knowledge Base' first.")
    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

def retrieve_context(question, k=5):
    vectorstore = _load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source_file', 'unknown')}]\n{d.page_content}"
        for d in docs
    )
    return context

def generate_rag_answer(question, context, chat_history=""):
    llm = _get_llm()
    prompt = f"""
You are an AI College Academic Assistant.

Student profile and conversation may be provided below.
Conversation:
{chat_history}

Answer the student's question using ONLY the supplied college knowledge-base context when the question is about college-specific facts.

Knowledge-base context:
{context}

Rules:
1. Do not invent college-specific facts.
2. If the context does not contain the answer, clearly say that the information is not available in the current knowledge base.
3. For general academic explanations, you may explain using your general knowledge, but clearly distinguish it from college-specific information.
4. Keep the answer clear and student-friendly.
5. If sources are present, mention the source file names.

Student question:
{question}
"""
    return llm.invoke(prompt).content

def ask_rag(question, comparison_only=False):
    llm = _get_llm()

    basic_prompt = f"""
You are a general-purpose AI assistant.

Answer the following question using your general knowledge.
Do NOT use the college knowledge base.
Do NOT assume any specific college's rules or policies.

Question:
{question}

Give a clear and useful answer. If the question asks about
college-specific rules, explain that a general answer cannot
confirm the specific college's requirements.
"""

    basic_response = llm.invoke(basic_prompt)

    if hasattr(basic_response, "content"):
        basic = basic_response.content
    else:
        basic = str(basic_response)

    if not basic or not basic.strip():
        basic = "The basic LLM did not return an answer."

    context = retrieve_context(question)
    rag = generate_rag_answer(question, context)

    return basic, rag

def summarize_document(content):
    llm = _get_llm()
    return llm.invoke(
        "Summarize the following college document into clear headings and bullet points. "
        "Do not add facts that are not in the document.\n\n" + content
    ).content
