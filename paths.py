import os
from pathlib import Path

# Base directory (parent of the folder where this file lives)
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_ROOT = Path(__file__).resolve().parent

ENV_FPATH = ROOT_DIR / ".env"

CODE_DIR = ROOT_DIR / "code"

APP_CONFIG_FPATH = APP_ROOT / "config" / "config.yaml"
PROMPT_CONFIG_FPATH = APP_ROOT / "config" / "prompt_config.yaml"
OUTPUTS_DIR = ROOT_DIR / "ITSMYGUIDE" / "outputs"

DATA_DIR = ROOT_DIR / "ITSMYGUIDE" / "Source"
PUBLICATION_FPATH = DATA_DIR / "publication.md"

VECTOR_DB_DIR = OUTPUTS_DIR / "vector_db"
CHAT_HISTORY_DB_FPATH = OUTPUTS_DIR / "chat_history.db"

# Chroma Cloud details
CHROMA_URLL = "chromadb-cloud-production.up.railway.app"
CHROMA_PORTT = "443"

# Weaviate Cloud details
WEAVIATE_URL = "http://8dzf8gehrqw15odh1q3ifa.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY="cUt4MXlab2hqNGllejFhTl81R2NjaVo0Y3d2eEhjczU3dmI5M0hESnRseE40VjI1Tk1FNUpUNGwrNXNnPV92MjAw"
CLASS_NAME = "Publication"
