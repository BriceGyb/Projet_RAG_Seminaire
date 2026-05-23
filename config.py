import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# Chargement des variables d'environnement
load_dotenv(dotenv_path=os.path.join("data", ".env"))

# Client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modèles
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"

# ChromaDB
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "juridique"

# Client ChromaDB persistant
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
