import os, dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from llm import llm
from neo4j_retriever import Neo4jFulltextRetriever, Neo4jBM25Retriever

dotenv.load_dotenv()

def build_qa(k: int = 10) -> RetrievalQA:
    """Devolve o objeto QA pronto para uso no Streamlit."""
    core_ret = Neo4jFulltextRetriever(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        k=k,
    )
    retriever = Neo4jBM25Retriever(n4j_ret=core_ret)

    template = """Você é um assistente jurídico que responde perguntas sobre o
        Acordo de Complementação Econômica nº 55 (ACE 55) entre Brasil e México.
        Use APENAS o contexto fornecido;

        Caso o usuário mande um cumprimento, responda apenas 'olá, como posso ajudar vc hj ? Sou especialista no acordo de complementação economica ace55.' -> Não cite textos ou trechos adicionais

        ### Contexto:
        {context}

        ### Pergunta:
        {question}

        Devolva a somente a resposta em formato de texto
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa
