from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import time
from dotenv import load_dotenv

load_dotenv()

_retriever = None

def load_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    
    # Check if faiss_index exists before loading
    if not os.path.exists("faiss_index/index.faiss"):
        raise ValueError("No document uploaded yet. Please upload a PDF first.")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    _retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    return _retriever

def ask_question(question: str):
    retriever = load_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
        max_tokens=512
    )

    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer ONLY using the context provided below.
Do NOT use any outside knowledge.
If the answer is not in the context, say exactly: 'This information is not in the document.'

Context:
{context}

Question: {question}

Answer:""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [doc.page_content[:200] for doc in docs]
    }

def ask_question_with_tracking(question: str):
    start = time.time()
    result = ask_question(question)
    response_time = time.time() - start

    # Only track with MLflow if running locally
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment("rag-chatbot")
            with mlflow.start_run():
                mlflow.log_param("question", question)
                mlflow.log_param("llm_model", "llama-3.3-70b-versatile")
                mlflow.log_metric("response_time_seconds", response_time)
        except Exception:
            pass  # Don't crash if MLflow unavailable

    return {
        "answer": result["answer"],
        "response_time": response_time
    }