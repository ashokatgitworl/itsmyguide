import os
import shutil
import zipfile
import requests
from typing import List,Annotated,Literal,Union,Dict,Any
from paths import CHROMA_URLL,CHROMA_PORTT
from langchain_core.tools import tool
import torch
import chromadb
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Annotated, List, Literal, Union
from typing_extensions import TypedDict
from utils import load_all_publications
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import ToolMessage,BaseMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from prompt_builder import build_prompt_from_config
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from utils import load_yaml_config


CHROMA_URL = CHROMA_URLL 
CHROMA_PORT = CHROMA_PORTT 
    

class AgentState(TypedDict):
    messages: List[BaseMessage]





def get_client():
    """
    Return a ChromaDB HTTP Client for Railway deployment.
    Railway ChromaDB must be deployed as:  python server.py
    """
    if not CHROMA_URL:
        raise ValueError("❌ CHROMA_URL is missing. Set it in Railway Variables.")

    # Remove scheme from host for chroma client
    host = CHROMA_URL.replace("https://", "").replace("http://", "")
    client = chromadb.HttpClient(
    host=CHROMA_URL, 
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

# def get_db_collection(collection_name: str = "publications"):
def get_db_collection(collection_name: str ):

    client = get_client()
    # return client.get_collection(name=collection_name)
    return client.get_or_create_collection(name=collection_name)


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


collection = get_db_collection(collection_name="publications")


@tool
def retrieve_relevant_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 0.5,
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        dict: Query results containing ids, documents, distances, and metadata
    """
    # logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    # logging.info("Embedding query...")
    query_embedding = embed_documents([query])[0]  # Get the first (and only) embedding
    # print(f"inside retrieve_relevant_documents tool with query step2 : {query_embedding}")

    # logging.info("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    # logging.info("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])
            # print(f"relevant_results inside tool: {relevant_results['documents']}")
    return relevant_results["documents"]


def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class CalcState(AgentState):
    """Graph State."""
    ops: Annotated[List[str], reduce_list]



# prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
# rag_assistant_prompt = prompt_config["rag_assistant_prompt"]
# @tool
# async def call_git_tool(tool_name: str, arguments: dict):
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:

#             # Initialize MCP session
#             await session.initialize()

#             # List available tools (optional but useful)
#             tools = await session.list_tools()
#             print("Available tools:", tools)

#             # Call a specific tool
#             result = await session.call_tool(
#                 name=tool_name,
#                 arguments=arguments
#             )

#             return result

def get_all_tools() -> List:
    """Return a list of all available tools."""
    return [
        retrieve_relevant_documents,
    ]