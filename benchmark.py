import json
from rag import answer_with_rag
from config import client, CHAT_MODEL

QA_PATH = "data/qa_pairs.json"
RESULTS_PATH = "benchmark_results.json"

PROMPT_JUGE = """Tu es un juge expert en droit. Évalue la qualité de la réponse générée par rapport à la réponse attendue.

Question posée : {question}

Réponse attendue : {expected_answer}

Réponse générée : {generated_answer}

Évalue la réponse générée sur trois critères, chacun noté de 1 à 5 :
- fidelite    : la réponse est-elle ancrée dans le contexte juridique et conforme aux sources ?
                (1 = inventée ou hors sujet, 5 = parfaitement fidèle au droit applicable)
- pertinence  : répond-elle précisément à la question posée ?
                (1 = complètement hors sujet, 5 = ciblée et directe)
- completude  : les éléments importants de la réponse attendue sont-ils tous couverts ?
                (1 = très incomplète, 5 = rien ne manque)

Retourne uniquement un objet JSON valide avec ce format exact :
{{
  "fidelite": <entier 1-5>,
  "pertinence": <entier 1-5>,
  "completude": <entier 1-5>,
  "justification": "<une phrase concise expliquant les scores>"
}}"""


def charger_qa_pairs(chemin: str) -> list:
    """Charge les paires Q&R depuis le fichier JSON."""
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)["qa_pairs"]


def juger_reponse(question: str, expected_answer: str, generated_answer: str) -> dict:
    """Envoie la question et les deux réponses au LLM-juge et retourne les scores."""
    prompt = PROMPT_JUGE.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

    reponse = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    return json.loads(reponse.choices[0].message.content)


def benchmark():
    """Point d'entrée principal du benchmark LLM-as-a-judge."""
    print("=" * 55)
    print("  BENCHMARK LLM-AS-A-JUDGE — RAG JURIDIQUE")
    print("=" * 55)

    qa_pairs = charger_qa_pairs(QA_PATH)
    print(f"\n{len(qa_pairs)} paires Q&R chargées depuis {QA_PATH}.\n")

    details = []
    total_fidelite = 0
    total_pertinence = 0
    total_completude = 0

    for i, qa in enumerate(qa_pairs, start=1):
        print(f"[{i:02d}/{len(qa_pairs)}] {qa['question'][:65]}...")

        # Génération de la réponse via le pipeline RAG
        rag_result = answer_with_rag(qa["question"])
        reponse_generee = rag_result["answer"]

        # Évaluation par le LLM-juge
        scores = juger_reponse(
            question=qa["question"],
            expected_answer=qa["expected_answer"],
            generated_answer=reponse_generee,
        )

        total_fidelite += scores.get("fidelite", 0)
        total_pertinence += scores.get("pertinence", 0)
        total_completude += scores.get("completude", 0)

        print(
            f"         Fidélité : {scores.get('fidelite')}/5 | "
            f"Pertinence : {scores.get('pertinence')}/5 | "
            f"Complétude : {scores.get('completude')}/5"
        )
        print(f"         {scores.get('justification', '')}\n")

        details.append({
            "id": qa["id"],
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "generated_answer": reponse_generee,
            "sources": rag_result["sources"],
            "scores": scores,
        })

    # Calcul des moyennes
    n = len(qa_pairs)
    moyennes = {
        "fidelite_moyenne":   round(total_fidelite / n, 2),
        "pertinence_moyenne": round(total_pertinence / n, 2),
        "completude_moyenne": round(total_completude / n, 2),
        "score_global_moyen": round((total_fidelite + total_pertinence + total_completude) / (3 * n), 2),
    }

    print("=" * 55)
    print("  RÉSUMÉ DES SCORES MOYENS (sur 5)")
    print("=" * 55)
    print(f"  Fidélité moyenne    : {moyennes['fidelite_moyenne']}/5")
    print(f"  Pertinence moyenne  : {moyennes['pertinence_moyenne']}/5")
    print(f"  Complétude moyenne  : {moyennes['completude_moyenne']}/5")
    print(f"  Score global moyen  : {moyennes['score_global_moyen']}/5")
    print("=" * 55)

    # Sauvegarde
    output = {"moyennes": moyennes, "details": details}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans {RESULTS_PATH}")


if __name__ == "__main__":
    benchmark()
