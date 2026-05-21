import os
from pathlib import Path
from config import client, collection, EMBEDDING_MODEL
from utils import load_pdf_text, clean_text, chunk_text

# Dossier racine contenant les PDFs (sous-dossiers inclus)
LOIS_DIR = os.path.join("data", "lois")

# Taille du lot pour les appels d'embedding
BATCH_SIZE = 50


def get_sources_deja_ingeres() -> set:
    """Récupère les noms de fichiers déjà présents dans la collection ChromaDB."""
    resultat = collection.get(include=["metadatas"])
    sources = set()
    for meta in resultat.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sources


def ingerer_pdf(chemin_pdf: str, sources_existantes: set) -> int:
    """
    Ingère un seul PDF dans ChromaDB.
    Retourne le nombre de chunks ajoutés (0 si déjà présent ou vide).
    """
    nom_fichier = os.path.basename(chemin_pdf)

    if nom_fichier in sources_existantes:
        print(f"  [IGNORÉ]    {nom_fichier} — déjà présent dans la collection.")
        return 0

    print(f"  [INGESTION] {nom_fichier}...")

    texte = load_pdf_text(chemin_pdf)
    texte = clean_text(texte)
    chunks = chunk_text(texte)

    if not chunks:
        print(f"  [AVERTISSEMENT] Aucun contenu extrait de {nom_fichier}.")
        return 0

    # Génération des embeddings par lots
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        lot = chunks[i:i + BATCH_SIZE]
        reponse = client.embeddings.create(input=lot, model=EMBEDDING_MODEL)
        embeddings.extend([r.embedding for r in reponse.data])

    # Stockage dans ChromaDB
    ids = [f"{nom_fichier}_chunk_{i}" for i in range(len(chunks))]
    metadonnees = [{"source": nom_fichier, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadonnees,
    )

    print(f"  [OK]        {len(chunks)} chunks ajoutés depuis {nom_fichier}.")
    return len(chunks)


def ingerer_tous():
    """Parcourt récursivement data/lois/ et ingère chaque PDF non encore présent."""
    print("=" * 55)
    print("  INGESTION DES LOIS — RAG JURIDIQUE")
    print("=" * 55)

    sources_existantes = get_sources_deja_ingeres()
    print(f"\nSources déjà présentes dans ChromaDB : {len(sources_existantes)}")

    fichiers_pdf = sorted(Path(LOIS_DIR).rglob("*.pdf"))

    if not fichiers_pdf:
        print(f"\nAucun fichier PDF trouvé dans {LOIS_DIR}/")
        return

    print(f"Fichiers PDF détectés            : {len(fichiers_pdf)}\n")

    total_chunks = 0
    for chemin in fichiers_pdf:
        total_chunks += ingerer_pdf(str(chemin), sources_existantes)

    print("\n" + "=" * 55)
    print(f"  Ingestion terminée — {total_chunks} chunks ajoutés au total.")
    print("=" * 55)


if __name__ == "__main__":
    ingerer_tous()
