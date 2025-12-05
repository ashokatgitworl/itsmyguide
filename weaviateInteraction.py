import os
from typing import List
from paths import WEAVIATE_URL, WEAVIATE_API_KEY,CLASS_NAME
# embeddings & splitting
import torch
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Property, DataType, Configure
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# weaviate
import weaviate

# your utils (same as before)
from utils import load_all_publications

# ------------------------------------------
# CONFIG - set these in Railway variables
# ------------------------------------------

WEAVIATE_URL = WEAVIATE_URL 
WEAVIATE_API_KEY =WEAVIATE_API_KEY 
CLASS_NAME = CLASS_NAME 


def get_client():
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY),
        )
        print("✅ Connected to WCD successfully!")
        return client
    except Exception as e:
        raise RuntimeError(f"❌ Failed to connect to WCD. Error: {e}")


# -----------------------------
# INIT COLLECTION (Class)
# -----------------------------
def initialize_schema():
    client = get_client()
    if client.collections.exists("Publication"):
        return
    
    client.collections.create(
        name="Publication",
        vector_config=Configure.Vectorizer.text2vec_openai(),  # New syntax
        properties=[
            Property(
                name="text",
                data_type=DataType.TEXT
            )
        ]
    )
    client.close()


def get_db_collection(collection_name: str = "publications"):
    client = get_client()
    # return client.get_collection(name=collection_name)
    return client.collections.get(collection_name)

# -----------------------------
# SPLITTING
# -----------------------------
def chunk_publication(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


# -----------------------------
# EMBEDDINGS
# -----------------------------
def embed_documents(documents: list[str]):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    return model.embed_documents(documents)



# -----------------------------
# INSERT DOCUMENTS
# -----------------------------
def insert_publications():
    client = get_client()
    collection = client.collections.get("Publication")

    publications = load_all_publications()

    with collection.batch.dynamic() as batch:
        for publication in publications:
            chunks = chunk_publication(publication)
            embeddings = embed_documents(chunks)

            for text_chunk, emb in zip(chunks, embeddings):
                batch.add_object(
                    properties={
                        "text": text_chunk
                    },
                    vector=emb  # vector must be passed separately
                )

# -----------------------------
# MAIN
# -----------------------------
def main():
    initialize_schema()
    insert_publications()


if __name__ == "__main__":
    main()