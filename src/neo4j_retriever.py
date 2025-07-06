# retriever.py
from neo4j import GraphDatabase
from langchain.schema import BaseRetriever, Document
from typing import List
from langchain.schema import BaseRetriever, Document
from pydantic import Field


class Neo4jFulltextRetriever:
    def __init__(self, uri, user, password,
                 index="chunk_text_idx", k=4):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.index  = index
        self.k      = k

    def get_relevant_documents(self, query: str):
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
        MATCH (node)<-[:HAS_CHUNK]-(d:Document)<-[:HAS_DOCUMENT]-(p:Protocol)
        RETURN node.text  AS text,
               node.page  AS page,
               d.file      AS file,
               p.number    AS protocol,
               score
        ORDER BY score ASC
        LIMIT $k
        """
        with self.driver.session() as s:
            res = s.run(cypher, index=self.index, q=query, k=self.k)
            docs = []
            for r in res:
                docs.append(
                    {
                     "page":     r["page"],
                     "file":     r["file"],
                     "protocol": r["protocol"],
                     "score":    r["score"],
                     "text":     r["text"]
                    }
                )
            return docs

class Neo4jBM25Retriever(BaseRetriever):
    """Adaptador que transforma Neo4jFulltextRetriever em BaseRetriever."""
    n4j_ret: Neo4jFulltextRetriever = Field(...)

    def __init__(self, n4j_ret: Neo4jFulltextRetriever, **data):
        super().__init__(n4j_ret=n4j_ret, **data)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = []
        for d in self.n4j_ret.get_relevant_documents(query):
            meta = {k: d[k] for k in ("page", "file", "protocol", "score")}
            docs.append(Document(page_content=d["text"], metadata=meta))
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)