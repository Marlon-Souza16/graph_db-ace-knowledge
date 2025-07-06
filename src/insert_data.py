from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob, pathlib, os
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)
splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)

pdfs = glob.glob("./files/ACE_055_*.pdf")
print("PDFs encontrados:", len(pdfs))

for pdf in pdfs:
    loader  = PyPDFLoader(pdf, extract_images=False)
    pages   = loader.load_and_split()
    chunks  = splitter.split_documents(pages)

    file_name = pathlib.Path(pdf).name
    proto_num = int(file_name.split("_")[2])   # 007 → 7 (truque rápido)

    with driver.session() as session:
        # 4.1 – garante Protocol e Document
        session.run("""
        MERGE (p:Protocol {number:$n})
        MERGE (d:Document {file:$file})
        MERGE (p)-[:HAS_DOCUMENT]->(d)
        """, n=proto_num, file=file_name)

        # 4.2 – insere Chunks
        for ch in chunks:
            res = session.run("""
            MATCH (d:Document {file:$file})
            CREATE (c:Chunk {
              text:$text,
              page:$page,
              tokens:$tokens
            })
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            file=file_name,
            text=ch.page_content,
            page=ch.metadata.get("page",0)+1,
            tokens=len(ch.page_content.split())
            )
        print(f" → {file_name}: {len(chunks)} chunks criados")
