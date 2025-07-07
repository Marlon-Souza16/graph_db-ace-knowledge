# Chatbot ACE-55 (Neo4j + LLM)

Este repositório contém um chatbot em português capaz de responder perguntas sobre o **Acordo de Complementação Econômica nº 55 (ACE-55)** assinado entre Brasil e México.  
O sistema emprega a técnica de *Retrieval-Augmented Generation* (RAG): os trechos relevantes do acordo são buscados em um grafo Neo4j, reranqueados com BM25 e, por fim, passados como contexto a um LLM (**Mistral 7 B**) rodando localmente.  

## Alunos

- Lucas Willian Serpa;
- Marlon de Souza;
- Ryan Bromati;


## Como funciona

1. **Ingestão de PDFs**  
   Três arquivos de teste (pasta `files/`) são fatiados em *chunks* de ~350 tokens com o `RecursiveCharacterTextSplitter`. Cada trecho é armazenado em Neo4j como nó `:Chunk`, ligado ao nó `:Document` e, indiretamente, ao protocolo correspondente (`:Protocol`).

2. **Busca híbrida**  
   O `Neo4jFulltextRetriever` faz o *recall* inicial; em seguida o `Neo4jBM25Retriever` realiza o reranqueamento para maior precisão.

3. **Geração de resposta**  
   O `RetrievalQA` (LangChain) reúne contexto + pergunta e aciona o **Mistral** via *endpoint* Ollama (ou container equivalente). A Streamlit (`app.py`) exibe a resposta e lista as páginas citadas em um *accordion* de “Fontes”.

## Principais componentes e dependências

- **Python ≥ 3.10**, `streamlit`, `langchain`, `neo4j-driver`, `python-dotenv`, `PyPDFLoader`.  
- **Neo4j 5.x** com índice *full-text* habilitado; não requer plugins além do **APOC** que já acompanha a edição Community.  
- **Mistral 7 B** servido localmente em `localhost:11434` (recomenda-se o Ollama).  
- Arquivos de configuração `.env` para credenciais do banco e URL do LLM.

Todas as dependências estão listadas em `requirements.txt`.

## Limitações atuais

Apesar de funcional, o chatbot cobre apenas três PDFs de teste, não utiliza embeddings vetoriais e não possui mecanismos de detecção de alucinação ou moderação de conteúdo. A execução do Mistral requer ~8 GB de RAM livre ou GPU compatível.