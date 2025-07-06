from langchain_community.llms import Ollama

llm = Ollama(
    model="mistral",
    temperature=0.3,
    top_p=0.9,
    base_url="http://localhost:11434"
)
