import os
import json
from pathlib import Path
from config import client, collection, EMBEDDING_MODEL
from utils import extract_articles_from_pdf, chunk_text, load_pdf_text, clean_text

LOIS_DIR = os.path.join("data", "lois")
STRUCTURED_DIR = os.path.join("data", "structured")
BATCH_SIZE = 50
MAX_ARTICLE_LENGTH = 1500  # sous-découpe les articles trop longs


def get_sources_deja_ingeres() -> set:
    resultat = collection.get(include=["metadatas"])
    sources = set()
    for meta in resultat.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sources


def sauvegarder_json(law_name: str, articles: list) -> str:
    """Sauvegarde les articles structurés dans data/structured/<law>.json."""
    os.makedirs(STRUCTURED_DIR, exist_ok=True)
    chemin = os.path.join(STRUCTURED_DIR, f"{law_name}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"law": law_name, "nb_articles": len(articles), "articles": articles},
                  f, ensure_ascii=False, indent=2)
    return chemin


def preparer_chunks(articles: list) -> list:
    """
    Transforme les articles en chunks prêts à être embarqués.
    Les articles dépassant MAX_ARTICLE_LENGTH sont sous-découpés
    tout en conservant leurs métadonnées (article_id, page).
    """
    chunks = []
    for art in articles:
        texte = art["text"]
        if len(texte) <= MAX_ARTICLE_LENGTH:
            chunks.append({
                "text": texte,
                "law": art["law"],
                "article_id": art["article_id"],
                "page": art["page"],
                "sub_chunk": 0,
            })
        else:
            sous_chunks = chunk_text(texte, chunk_size=MAX_ARTICLE_LENGTH, overlap=120)
            for i, sc in enumerate(sous_chunks):
                chunks.append({
                    "text": sc,
                    "law": art["law"],
                    "article_id": art["article_id"],
                    "page": art["page"],
                    "sub_chunk": i,
                })
    return chunks


def ingerer_pdf(chemin_pdf: str, sources_existantes: set) -> int:
    nom_fichier = os.path.basename(chemin_pdf)
    law_name = os.path.splitext(nom_fichier)[0]

    if nom_fichier in sources_existantes:
        print(f"  [IGNORÉ]     {nom_fichier} — déjà présent dans la collection.")
        return 0

    print(f"  [EXTRACTION] {nom_fichier}...")
    articles = extract_articles_from_pdf(chemin_pdf)

    if not articles:
        print(f"  [FALLBACK]   Aucun article détecté dans {nom_fichier} — découpage par chunks.")
        texte = clean_text(load_pdf_text(chemin_pdf))
        raw_chunks = chunk_text(texte)
        if not raw_chunks:
            print(f"  [AVERTISSEMENT] Aucun contenu extrait de {nom_fichier}.")
            return 0
        articles = [
            {"law": law_name, "article_id": str(i), "page": 0, "text": c}
            for i, c in enumerate(raw_chunks)
        ]

    chemin_json = sauvegarder_json(law_name, articles)
    print(f"  [JSON]       {len(articles)} articles → {chemin_json}")

    chunks = preparer_chunks(articles)
    print(f"  [CHUNKS]     {len(chunks)} chunks à embarquer...")

    textes = [c["text"] for c in chunks]
    embeddings = []
    for i in range(0, len(textes), BATCH_SIZE):
        lot = textes[i:i + BATCH_SIZE]
        reponse = client.embeddings.create(input=lot, model=EMBEDDING_MODEL)
        embeddings.extend([r.embedding for r in reponse.data])

    ids = [
        f"{law_name}_art{c['article_id']}_idx{i}_sub{c['sub_chunk']}"
        for i, c in enumerate(chunks)
    ]
    metadonnees = [
        {
            "source": nom_fichier,
            "law": c["law"],
            "article_id": c["article_id"],
            "page": c["page"],
            "sub_chunk": c["sub_chunk"],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=textes,
        metadatas=metadonnees,
    )

    print(f"  [OK]         {len(chunks)} chunks ajoutés depuis {nom_fichier}.")
    return len(chunks)


def ingerer_tous():
    print("=" * 55)
    print("  INGESTION STRUCTURÉE — RAG JURIDIQUE")
    print("=" * 55)

    sources_existantes = get_sources_deja_ingeres()
    print(f"\nSources déjà présentes dans ChromaDB : {len(sources_existantes)}")

    fichiers_pdf = sorted(Path(LOIS_DIR).rglob("*.pdf"))

    if not fichiers_pdf:
        print(f"\nAucun fichier PDF trouvé dans {LOIS_DIR}/")
        return

    print(f"Fichiers PDF détectés              : {len(fichiers_pdf)}\n")

    total_chunks = 0
    for chemin in fichiers_pdf:
        total_chunks += ingerer_pdf(str(chemin), sources_existantes)

    print("\n" + "=" * 55)
    print(f"  Ingestion terminée — {total_chunks} chunks ajoutés au total.")
    print("=" * 55)


if __name__ == "__main__":
    ingerer_tous()
