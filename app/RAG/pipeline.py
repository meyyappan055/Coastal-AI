from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from RAG.config import DOCS_PATH


def load_documents():
    documents = []
    for pdf_path in DOCS_PATH.glob("**/*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load_and_split())
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks


def get_vectorstore(docs):
    for doc in docs:
        doc.page_content = f"passage: {doc.page_content}"

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    persist_dir = "./chroma_langchain_db"
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    for i in range(0, len(docs), 100):
        batch = docs[i:i+100]
        vector_store.add_documents(batch)

    return vector_store

def get_llm():
    return ChatOllama(model="llama3.1")



def retrieve_answer(vector_store, llm, prompt, question, k=3):
    question = f"query: {question}"
    retrieved_docs = vector_store.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=30,
        lambda_mult=0.7
    )
  
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i+1}] {doc.page_content[:200]}...\n")

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    messages = prompt.format_messages(context=context, question=question) 
    response = llm.invoke(messages)

    return response.content, context
