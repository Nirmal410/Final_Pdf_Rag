
import streamlit as st


def display_sources(sources):

    if not sources:
        return

    st.markdown("**Sources**")

    for source in sources:

        st.caption(
            f"📄 {source['source']} "
            f"| Page {source['page']}"
        )


def display_chat_history():

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):
                display_sources(message["sources"])


def display_retrieved_context(documents):

    if not documents:
        return

    with st.expander("View retrieved context"):

        for index, document in enumerate(
            documents, start=1
        ):

            st.markdown(f"**Chunk {index}**")

            st.write(document["text"])

            st.caption(
                f"Source: {document['metadata']['source']} "
                f"| Page: {document['metadata']['page']}"
            )
