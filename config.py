import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


def _get_api_key() -> str:
    # 1. Variable d'environnement déjà présente
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # 2. Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # 3. Fichier .env local
    load_dotenv(dotenv_path=os.path.join("data", ".env"))
    return os.getenv("OPENAI_API_KEY")


# Modèles
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"

# Client OpenAI
client = OpenAI(api_key=_get_api_key())

# ChromaDB
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "juridique"

# Client ChromaDB persistant
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
