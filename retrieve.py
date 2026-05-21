from config import client, collection, EMBEDDING_MODEL


def search(query: str, top_k: int = 3) -> dict:
    """
    Encode la requête et retourne les top_k chunks les plus proches dans ChromaDB.
    Retourne un dict avec clés : documents, metadatas, distances.
    """
    # Embedding de la requête
    reponse = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    embedding_requete = reponse.data[0].embedding

    # Recherche vectorielle
    resultats = collection.query(
        query_embeddings=[embedding_requete],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "documents": resultats["documents"][0],
        "metadatas": resultats["metadatas"][0],
        "distances": resultats["distances"][0],
    }


if __name__ == "__main__":
    requete = "Quelles sont les conditions de validité d'un contrat selon le Code civil du Québec ?"
    resultats = search(requete, top_k=3)

    print(f"Requête : {requete}\n")
    for i, (doc, meta, dist) in enumerate(
        zip(resultats["documents"], resultats["metadatas"], resultats["distances"])
    ):
        print(f"--- Chunk {i + 1} (distance : {dist:.4f}) ---")
        print(f"Source : {meta.get('source', 'N/A')} | Chunk #{meta.get('chunk_index', 'N/A')}")
        print(f"{doc[:300]}...")
        print()
