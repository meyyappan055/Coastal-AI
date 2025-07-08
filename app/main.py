from RAG.pipeline import load_documents, chunk_documents, get_vectorstore, get_llm, retrieve_answer
from RAG.utils import get_prompt_template

question = "frequency of Noctiluca blooms is high in which months??"

documents = load_documents()
chunks = chunk_documents(documents)

vector_store = get_vectorstore(chunks)

llm = get_llm()
prompt = get_prompt_template()

answer, context = retrieve_answer(vector_store, llm, prompt, question)

# print("Context:\n", context)
print("\nAnswer:", answer)
