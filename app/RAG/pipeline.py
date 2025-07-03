from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from RAG.config import COHERE_KEY, PDF_PATH

def load_documents():
    loader = PyPDFLoader(str(PDF_PATH))
    return loader.load_and_split()

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def get_vectorstore(docs):
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_KEY)
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    vector_store.add_documents(docs)
    return vector_store

def get_llm():
    return ChatOllama(model="llama3.1")

def retrieve_answer(vector_store, llm, prompt, question, k=2):
    retrieved_docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content, context
