from langchain.schema import BaseRetriever, Document
from typing import List

class Neo4jBM25Retriever(BaseRetriever):
    """Envolve Neo4jFulltextRetriever para atender ao contrato BaseRetriever."""
    def __init__(self, n4j_ret):
        self.n4j_ret = n4j_ret

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = []
        for d in self.n4j_ret.get_relevant_documents(query):
            meta = {k: d[k] for k in ("page", "file", "protocol", "score")}
            docs.append(Document(page_content=d["text"], metadata=meta))
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
