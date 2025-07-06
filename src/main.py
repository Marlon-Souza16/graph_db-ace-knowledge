# main.py – RAG completo
import os, dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from llm import llm

dotenv.load_dotenv()

from neo4j_retriever import Neo4jFulltextRetriever, Neo4jBM25Retriever

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

core_ret = Neo4jFulltextRetriever(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    k=4
)

print(NEO4J_URI)

retriever = Neo4jBM25Retriever(n4j_ret=core_ret)

TEMPLATE = """Você é um assistente jurídico que responde perguntas sobre o
Acordo de Complementação Econômica nº 55 (ACE 55) entre Brasil e México.
Use APENAS o contexto fornecido. Se não estiver no contexto, responda "não encontrei".

### Contexto:
{context}

### Pergunta:
{question}

### Resposta (cite arquivo e página se possível):
"""
prompt = PromptTemplate(template=TEMPLATE,
                        input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
)

if __name__ == "__main__":
    question = "Qual era o ICR mínimo para veículos a gasolina no Protocolo de 2016?"
    result   = qa(question)

    print("\n🤖 Resposta:\n", result["result"])
    print("\n--- Fontes ---")
    for doc in result["source_documents"]:
        m = doc.metadata
        print(f"- {m['file']} pág.{m['page']} (Protocolo {m['protocol']}) "
              f"score {m['score']:.2f}")
