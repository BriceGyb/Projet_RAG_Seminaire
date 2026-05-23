import json
from retrieve import search

QA_PATH = "data/qa_pairs.json"
RESULTS_PATH = "eval_results.json"


def charger_qa_pairs(chemin: str) -> list:
    """Charge les paires Q&R depuis le fichier JSON."""
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)["qa_pairs"]


def est_pertinent(document: str, keywords: list, seuil: int = 2) -> bool:
    """
    Juge un chunk pertinent si au moins 'seuil' keywords y apparaissent.
    La comparaison est insensible à la casse.
    """
    doc_min = document.lower()
    correspondances = sum(1 for kw in keywords if kw.lower() in doc_min)
    return correspondances >= seuil


def calculer_metriques(qa_pairs: list, k: int) -> dict:
    """
    Calcule Recall@k, Precision@k et MRR@k sur l'ensemble des questions.
    Un chunk est jugé pertinent si au moins 2 keywords sont présents.
    """
    recalls, precisions, rr_scores = [], [], []

    for qa in qa_pairs:
        question = qa["question"]
        keywords = qa["relevant_chunk_keywords"]

        resultats = search(question, top_k=k)
        documents = resultats["documents"]

        pertinences = [est_pertinent(doc, keywords) for doc in documents]
        n_pertinents = sum(pertinences)

        # Recall@k : au moins un chunk pertinent retrouvé parmi les k
        recalls.append(1.0 if n_pertinents > 0 else 0.0)

        # Precision@k : proportion de chunks pertinents parmi les k récupérés
        precisions.append(n_pertinents / k)

        # MRR@k : inverse du rang du premier chunk pertinent
        rr = 0.0
        for rang, pertinent in enumerate(pertinences, start=1):
            if pertinent:
                rr = 1.0 / rang
                break
        rr_scores.append(rr)

    n = len(qa_pairs)
    return {
        f"Recall@{k}":    round(sum(recalls) / n, 4),
        f"Precision@{k}": round(sum(precisions) / n, 4),
        f"MRR@{k}":       round(sum(rr_scores) / n, 4),
    }


def evaluer():
    """Point d'entrée principal de l'évaluation du retrieval."""
    print("=" * 55)
    print("  ÉVALUATION DU RETRIEVAL — RAG JURIDIQUE")
    print("=" * 55)

    qa_pairs = charger_qa_pairs(QA_PATH)
    print(f"\n{len(qa_pairs)} paires Q&R chargées depuis {QA_PATH}.\n")

    tous_resultats = {}

    for k in [1, 3, 5]:
        print(f"Calcul des métriques pour k = {k}...")
        metriques = calculer_metriques(qa_pairs, k)
        tous_resultats[f"k={k}"] = metriques
        for nom, val in metriques.items():
            print(f"  {nom:<14} : {val}")
        print()

    # Tableau récapitulatif
    print("=" * 52)
    print(f"{'k':<5} {'Recall@k':<14} {'Precision@k':<14} {'MRR@k':<10}")
    print("-" * 52)
    for k in [1, 3, 5]:
        m = tous_resultats[f"k={k}"]
        print(
            f"{k:<5} "
            f"{m[f'Recall@{k}']:<14} "
            f"{m[f'Precision@{k}']:<14} "
            f"{m[f'MRR@{k}']:<10}"
        )
    print("=" * 52)

    # Sauvegarde
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(tous_resultats, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans {RESULTS_PATH}")


if __name__ == "__main__":
    evaluer()
