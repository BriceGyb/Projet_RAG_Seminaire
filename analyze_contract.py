import sys
import json
from config import client, CHAT_MODEL
from retrieve import search
from utils import load_pdf_text, clean_text


PROMPT_EXTRACTION_CLAUSES = """Tu es un juriste expert. Analyse ce contrat et extrais ses clauses importantes.

Contrat :
{contract_text}

Retourne un objet JSON valide avec ce format exact :
{{
  "type_contrat": "<type du contrat ex: bail, vente, travail, service...>",
  "parties": ["<partie 1>", "<partie 2>"],
  "clauses": [
    {{
      "titre": "<titre court de la clause>",
      "texte": "<texte exact ou résumé fidèle de la clause>",
      "importance": "<haute|moyenne|faible>"
    }}
  ]
}}

Extrais toutes les clauses importantes (5 à 15 clauses)."""


PROMPT_ANALYSE_CLAUSE = """Tu es un assistant juridique expert en droit québécois et canadien.

Contexte légal applicable :
{legal_context}

Clause du contrat à analyser :
Titre : {clause_titre}
Texte : {clause_texte}

Analyse cette clause sur trois points :
1. Validité légale : est-elle conforme aux lois applicables ?
2. Risques : quels risques présente-t-elle pour chaque partie ?
3. Articles applicables : quels articles de loi s'appliquent ?

Sois précis et concis."""


PROMPT_QA_CONTRAT = """Tu es un assistant juridique expert. Tu as analysé un contrat et tu réponds aux questions de l'utilisateur.

Type de contrat : {type_contrat}
Parties : {parties}

Clauses du contrat :
{clauses_resume}

Contexte légal pertinent :
{legal_context}

Question : {question}

Réponds de manière formelle et précise en te basant sur le contrat et le contexte légal ci-dessus."""


def extraire_clauses(contract_text: str) -> dict:
    """Utilise le LLM pour extraire les clauses structurées du contrat."""
    # Tronque si trop long (limite de contexte)
    texte_tronque = contract_text[:12000]

    reponse = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_EXTRACTION_CLAUSES.format(contract_text=texte_tronque)
        }],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(reponse.choices[0].message.content)


def analyser_clause(clause: dict) -> dict:
    """Analyse une clause en la croisant avec le corpus juridique via RAG."""
    query = f"{clause['titre']} {clause['texte']}"
    resultats = search(query, top_k=3)

    legal_context = "\n\n---\n\n".join([
        f"[{meta.get('source', '?')} art.{meta.get('article_id', '?')} p.{meta.get('page', '?')}]\n{doc}"
        for doc, meta in zip(resultats["documents"], resultats["metadatas"])
    ])

    reponse = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_ANALYSE_CLAUSE.format(
                legal_context=legal_context,
                clause_titre=clause["titre"],
                clause_texte=clause["texte"],
            )
        }],
        temperature=0.1,
    )

    return {
        "titre": clause["titre"],
        "importance": clause["importance"],
        "texte_original": clause["texte"],
        "analyse": reponse.choices[0].message.content,
        "sources_legales": [
            f"{meta.get('source', '?')} art.{meta.get('article_id', '?')}"
            for meta in resultats["metadatas"]
        ],
    }


def afficher_rapport(structure: dict, analyses: list):
    """Affiche le rapport d'analyse dans la console."""
    print("\n" + "=" * 60)
    print(f"  ANALYSE DE CONTRAT — {structure.get('type_contrat', '').upper()}")
    print("=" * 60)
    print(f"  Parties : {' / '.join(structure.get('parties', []))}")
    print(f"  Clauses analysées : {len(analyses)}")
    print("=" * 60)

    for i, analyse in enumerate(analyses, 1):
        importance = analyse["importance"].upper()
        print(f"\n[{i}] {analyse['titre']}  [{importance}]")
        print("-" * 50)
        print(analyse["analyse"])
        print(f"\nSources : {', '.join(analyse['sources_legales'])}")

    print("\n" + "=" * 60)


def repondre_question(question: str, structure: dict, analyses: list) -> str:
    """Répond à une question sur le contrat en croisant contenu du contrat et corpus légal."""
    clauses_resume = "\n".join([
        f"- {a['titre']} [{a['importance']}] : {a['texte_original'][:200]}"
        for a in analyses
    ])

    resultats = search(question, top_k=3)
    legal_context = "\n\n---\n\n".join([
        f"[{meta.get('source', '?')} art.{meta.get('article_id', '?')}]\n{doc}"
        for doc, meta in zip(resultats["documents"], resultats["metadatas"])
    ])

    reponse = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_QA_CONTRAT.format(
                type_contrat=structure.get("type_contrat", "inconnu"),
                parties=", ".join(structure.get("parties", [])),
                clauses_resume=clauses_resume,
                legal_context=legal_context,
                question=question,
            )
        }],
        temperature=0.1,
    )
    return reponse.choices[0].message.content


def mode_qa(structure: dict, analyses: list):
    """Mode interactif CLI : questions-réponses sur le contrat analysé."""
    print("\n  MODE Q&R — Posez vos questions sur ce contrat.")
    print("  (tapez 'quitter' pour terminer)\n")

    while True:
        question = input("Votre question : ").strip()
        if not question or question.lower() in ("quitter", "exit", "q"):
            print("Au revoir.")
            break
        print(f"\nRéponse :\n{repondre_question(question, structure, analyses)}\n")


def analyser_contrat(chemin_pdf: str):
    print(f"\nChargement du contrat : {chemin_pdf}")
    texte = clean_text(load_pdf_text(chemin_pdf))

    if not texte.strip():
        print("Erreur : aucun texte extrait du PDF.")
        return

    print("Extraction des clauses en cours...")
    structure = extraire_clauses(texte)

    clauses = structure.get("clauses", [])
    print(f"{len(clauses)} clauses détectées. Analyse juridique en cours...\n")

    analyses = []
    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['titre']}...")
        analyses.append(analyser_clause(clause))

    afficher_rapport(structure, analyses)

    # Sauvegarde du rapport
    rapport = {"structure": structure, "analyses": analyses}
    chemin_rapport = chemin_pdf.replace(".pdf", "_analyse.json")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    print(f"\nRapport sauvegardé : {chemin_rapport}")

    mode_qa(structure, analyses)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python analyze_contract.py <chemin_vers_contrat.pdf>")
        sys.exit(1)

    analyser_contrat(sys.argv[1])
