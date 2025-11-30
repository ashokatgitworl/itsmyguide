import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FPATH = os.path.join(ROOT_DIR, ".env")

CODE_DIR = os.path.join(ROOT_DIR, "code")

APP_CONFIG_FPATH = os.path.join(ROOT_DIR, "ITSMYGUIDE/config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(ROOT_DIR, "ITSMYGUIDE/config", "prompt_config.yaml")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "ITSMYGUIDE/outputs")

DATA_DIR = os.path.join(ROOT_DIR, "ITSMYGUIDE\Source")
PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")

VECTOR_DB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")
CHAT_HISTORY_DB_FPATH = os.path.join(OUTPUTS_DIR, "chat_history.db")
CHROMA_URLL = "http://chromadb-cloud-production.up.railway.app"
CHROMA_PORTT = "443"