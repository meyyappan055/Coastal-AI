import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
pdf_path = BASE_DIR / "documents" / "reports" / "report1.pdf"

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found at: {pdf_path}")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
COHERE_KEY = os.getenv("COHERE_KEY")


if not GOOGLE_API_KEY or not LANGSMITH_API_KEY or not COHERE_KEY:
    print("Please set the  environment variables properly.")
    exit(1)


# #LLM
llm = ChatOllama(model="llama3.1")

#Embeddings - Cohere
embeddings = CohereEmbeddings(model="embed-english-v3.0" , cohere_api_key=COHERE_KEY)


#Vector Store - Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
print("Vector store created.")


# #load the data

loader = PyPDFLoader(str(pdf_path))
documents = loader.load_and_split()

async def load_pages_async(loader):
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

# # print("Loaded documents.")
# # print(f"{documents[0].metadata}\n")
# # print(documents[1].page_content)


# #Chunking (Later -> recursive splitter + semantic splitter)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       
    chunk_overlap=200,   
    separators=["\n\n", "\n", ".", " ", ""]
)

all_splits = recursive_splitter.split_documents(documents)

# print(f"Number of chunks: {len(all_splits)}")
# print(f"Chunk content: {( all_splits[0].page_content)}")


# # # embed and store
document_ids = vector_store.add_documents(documents=all_splits)
# print(document_ids[:3]) # ID of vector chunks in chroma


# RETREIVAL AND GENERATION

question = "What types of data does INCOIS collect and how is it distributed to users?"

retrieved_docs = vector_store.similarity_search(question, k=2) # Top 2 similar chunks

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a coastal research assistant trained to answer oceanographic and ecological questions based on given documents."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

messages = prompt_template.format_messages(context=context, question=question)
print(context)

response = llm.invoke(messages)

print("Answer:", response.content)

