from langchain_core.prompts import ChatPromptTemplate

def get_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a coastal research assistant trained to answer oceanographic and ecological questions based on given documents."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])