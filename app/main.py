import os
from langchain.chat_models import init_chat_model
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_KEY = os.getenv("LANGSMITH_KEY")


if not GEMINI_API_KEY or not LANGSMITH_KEY:
    print("Please set the GEMINI_API_KEY and LANGSMITH_KEY environment variables.")
    exit(1)


#LLM
llm = init_chat_model("gemini-1.5-pro", model_provider="google_genai")

#Embeddings - Cohere
embeddings = CohereEmbeddings(model="embed-english-v3.0")


#Vector Store - Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

