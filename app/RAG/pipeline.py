from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from RAG.config import COHERE_KEY, DOCS_PATH

try : 
    print("Loading documents...")
    def load_documents():
        documents = []
        for pdf_path in DOCS_PATH.glob("**/*.pdf"):
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load_and_split())
        return documents
except:
    print("Error loading documents")


try : 
    print("Chunking documents...")
    def chunk_documents(documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)
    
except:
    print("Error chunking documents")


try:
    def get_vectorstore(docs):
        for doc in docs:
            doc.page_content = f"passage: {doc.page_content}"
        
        embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        encode_kwargs={"normalize_embeddings": True}
        )

        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )
        vector_store.add_documents(docs)
        return vector_store
except:
    print("Error creating vectorstore")


def get_llm():
    return ChatOllama(model="llama3.1")


try: 
    def retrieve_answer(vector_store, llm, prompt, question, k=2):

        question = f"query: {question}"
        
        retrieved_docs = vector_store.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        return response.content, context

except:
    print("Error retrieving answer")