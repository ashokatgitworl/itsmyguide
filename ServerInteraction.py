import os
import torch
import chromadb
from paths import CHROMA_URLL,CHROMA_PORTT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_all_publications


# ------------------------------------------
# CONFIG FOR RAILWAY
# ------------------------------------------
# Railway injects PORT dynamically → mandatory for HTTP servers
# CHROMA_URL = "https://chromadb-cloud-production.up.railway.app" #os.getenv("chromadb-cloud-production.up.railway.app")  # example → http://my-chroma.up.railway.app
CHROMA_URL = CHROMA_URLL #os.getenv("CHROMA_URLL")  # example → http://my-chroma.up.railway.app
CHROMA_PORT = CHROMA_PORTT # int(os.getenv("CHROMA_PORT", "8080"))  # set this manually for client


def get_client():
    """
    Return a ChromaDB HTTP Client for Railway deployment.
    Railway ChromaDB must be deployed as:  python server.py
    """
    if not CHROMA_URL:
        raise ValueError("❌ CHROMA_URL is missing. Set it in Railway Variables.")

    # Remove scheme from host for chroma client
    host = CHROMA_URL.replace("https://", "").replace("http://", "")

    # client = chromadb.HttpClient(
    #     host=host,
    #     port=CHROMA_PORT,
    #     ssl=False  # Railway free-tier uses HTTP, not HTTPS
    # )
    client = chromadb.HttpClient(
    host=CHROMA_URL, #"https://chromadb-cloud-production.up.railway.app",
    port=CHROMA_PORT,
    ssl=True
)
    return client


def initialize_db(collection_name: str = "publications") -> chromadb.Collection:
    """Initialize a ChromaDB collection hosted on Railway."""
    client = get_client()

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection on Railway: {collection_name}")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            }
        )
        print(f"Created new collection on Railway: {collection_name}")

    return collection


def get_db_collection(collection_name: str = "publications"):
    client = get_client()
    return client.get_collection(name=collection_name)


def chunk_publication(publication: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(publication)


def embed_documents(documents: list[str]):
    """Embed text using HuggingFace embeddings."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    return model.embed_documents(documents)


def insert_publications(collection: chromadb.Collection, publications: list[str]):
    """Insert publication chunks + embeddings into Railway-hosted ChromaDB."""
    next_id = collection.count()

    for publication in publications:
        chunks = chunk_publication(publication)
        embeddings = embed_documents(chunks)

        ids = [
            f"document_{i}" for i in range(next_id, next_id + len(chunks))
        ]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
        )

        next_id += len(chunks)


def main():
    collection = initialize_db("publications")
    publications = load_all_publications()
    insert_publications(collection, publications)

    print(f"Total docs in Railway ChromaDB: {collection.count()}")


if __name__ == "__main__":
    main()
