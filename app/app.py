import streamlit as st
from RAG.pipeline import load_documents, chunk_documents, get_vectorstore, get_llm, retrieve_answer
from RAG.utils import get_prompt_template
from pathlib import Path

DOCS_PATH = Path("documents/reports/research_papers")

st.set_page_config(page_title="Coastal Research AI", layout="wide")
st.title("🌊 RAG powered Coastal Research Assistant")

st.sidebar.markdown("### 📄 Using PDFs from:")
st.sidebar.code(str(DOCS_PATH))

if "vector_store" not in st.session_state:
    with st.spinner("Processing existing PDFs..."):
        documents = load_documents()
        chunks = chunk_documents(documents)
        vector_store = get_vectorstore(chunks)
        llm = get_llm()
        prompt = get_prompt_template()

    st.session_state.vector_store = vector_store
    st.session_state.llm = llm
    st.session_state.prompt = prompt

    st.sidebar.success("✅ PDFs processed & indexed!")

question = st.text_input("❓ Ask a question")

if st.button("🔍 Get Answer"):
    with st.spinner("Retrieving answer..."):
        answer, docs = retrieve_answer(
            st.session_state.vector_store,
            st.session_state.llm,
            st.session_state.prompt,
            question
        )

    st.markdown(f"### ✅ Answer:\n{answer}")

    with st.expander("📜 Sources Used"):
        for i, doc in enumerate(docs):
            st.markdown(f"**{i+1}.** {doc.page_content[:300]}...")
