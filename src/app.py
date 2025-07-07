import streamlit as st
from main import build_qa

st.set_page_config(page_title="ACE-55 Chatbot", page_icon="📜")
st.title("📜 Chatbot ACE-55 (Neo4j + LLM)")

@st.cache_resource
def load_qa():
    return build_qa(k=10)

qa = load_qa()

user_input = st.chat_input("Digite sua pergunta sobre o ACE-55…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    result = qa(user_input)

    with st.chat_message("assistant"):
        st.markdown(result["result"])

        with st.expander("📑 Fontes"):
            for doc in result["source_documents"]:
                m = doc.metadata
                st.write(
                    f"• **{m['file']}**, pág. {m['page']} "
                    f"(Protocolo {m['protocol']}) — score {m['score']:.2f}"
                )
