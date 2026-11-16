
import hashlib
from pathlib import Path

import streamlit as st

from config.settings import UPLOAD_DIR

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.metadata import create_chunks

from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retrieval_service import RetrievalService
from src.pipeline.rag_pipeline import RAGPipeline

from src.ui.chat_interface import (
    display_sources,
    display_chat_history,
    display_retrieved_context
)


st.set_page_config(
    page_title="Local PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# Application heading
st.title("📚 Local PDF RAG Chatbot")

st.caption(
    "Ask questions about your PDF using "
    "local embeddings, hybrid retrieval and Ollama."
)


# Sidebar
with st.sidebar:

    st.header("Document Manager")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True,
        disabled=uploaded_file is None
    )

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    if st.session_state.pdf_name:

        st.success(
            f"Active PDF: {st.session_state.pdf_name}"
        )


# PDF processing
if process_button and uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    if len(pdf_bytes) > 50 * 1024 * 1024:

        st.error("PDF must be smaller than 50 MB.")

        st.stop()

    pdf_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    pdf_path = UPLOAD_DIR / Path(uploaded_file.name).name

    try:

        with st.spinner("Processing PDF..."):

            pdf_path.write_bytes(pdf_bytes)

            # Extract PDF text
            documents = load_pdf(str(pdf_path))

            # Create chunks
            chunks = create_chunks(documents)

            # Clear previous PDF collection
            store = ChromaStore()
            store.clear()

            # Build retrieval service
            retrieval_service = RetrievalService(chunks)

            # Build RAG pipeline
            pipeline = RAGPipeline(
                retrieval_service
            )

            # Save application state
            st.session_state.rag_pipeline = pipeline

            st.session_state.pdf_hash = pdf_hash

            st.session_state.pdf_name = uploaded_file.name

            st.session_state.messages = []

        st.success(
            f"Successfully processed {len(chunks)} chunks."
        )

    except Exception as error:

        st.session_state.rag_pipeline = None

        st.error(f"PDF processing failed: {error}")


# Chat interface
if st.session_state.rag_pipeline is None:

    st.info(
        "Upload a PDF and click Process PDF to start chatting."
    )

else:

    display_chat_history()

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):

            try:

                with st.spinner("Searching and generating answer..."):

                    # Previous messages, excluding current question
                    history = st.session_state.messages[:-1]

                    result = st.session_state.rag_pipeline.answer(
                        question,
                        history=history
                    )

                st.markdown(result["answer"])

                display_sources(result["sources"])

                display_retrieved_context(
                    result["retrieved_chunks"]
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

            except Exception as error:

                st.error(
                    f"Unable to generate answer: {error}"
                )
