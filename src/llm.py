from langchain_community.llms import Ollama

llm = Ollama(
    model="mistral",
    temperature=0.2,
    base_url="http://localhost:11434"
)
