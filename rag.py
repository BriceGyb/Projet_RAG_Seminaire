from config import client, CHAT_MODEL
from retrieve import search


def retrieve_context(query: str, top_k: int = 3) -> tuple:
    """Récupère les top_k chunks les plus pertinents pour la requête."""
    resultats = search(query, top_k=top_k)
    return resultats["documents"], resultats["metadatas"]


def answer_with_rag(question: str) -> dict:
    """
    Génère une réponse juridique à partir du contexte récupéré par RAG.
    Retourne la réponse, les sources et les chunks utilisés.
    """
    documents, metadonnees = retrieve_context(question)

    contexte = "\n\n---\n\n".join(documents)
    sources = list({
        f"{meta.get('source', 'N/A')} art.{meta.get('article_id', '?')} p.{meta.get('page', '?')}"
        for meta in metadonnees
    })

    prompt_systeme = (
        "Vous êtes un assistant juridique expert. "
        "Répondez uniquement à partir du contexte fourni, de manière formelle et précise. "
        "Si la réponse ne peut pas être déterminée à partir du contexte, indiquez-le explicitement."
    )

    prompt_utilisateur = (
        f"Contexte juridique :\n{contexte}\n\n"
        f"Question : {question}\n\n"
        "Répondez de manière formelle et précise en vous basant exclusivement sur le contexte ci-dessus."
    )

    reponse = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_utilisateur},
        ],
        temperature=0.1,
    )

    return {
        "answer": reponse.choices[0].message.content,
        "sources": sources,
        "context_chunks": documents,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  ASSISTANT JURIDIQUE — RAG")
    print("=" * 55)
    question = input("\nPosez votre question juridique : ").strip()

    if not question:
        print("Aucune question saisie.")
    else:
        print("\nRecherche en cours...\n")
        resultat = answer_with_rag(question)
        print(f"Réponse :\n{resultat['answer']}\n")
        print(f"Sources utilisées : {', '.join(resultat['sources'])}")
