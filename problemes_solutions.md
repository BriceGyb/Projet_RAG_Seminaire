# Difficultés Rencontrées et Solutions Apportées

---

## Problème 1 — Filtrage des questions hors-sujet dans le chatbot juridique

### Description du problème

Dans un assistant juridique basé sur RAG, chaque message de l'utilisateur déclenche un pipeline coûteux :

```
Question → Embedding (API OpenAI) → Recherche vectorielle (ChromaDB) → Génération (API OpenAI)
```

Sans filtre, des messages sans valeur juridique comme **"Bonjour"**, **"Merci"**, **"Ça va ?"** ou **"Parle-moi de football"** déclenchent ce pipeline complet inutilement. Cela entraîne :

- Des **coûts API inutiles** (appels d'embedding + appels LLM)
- Une **latence inutile** pour l'utilisateur
- Des **réponses incohérentes** : le système tente de répondre à "Bonjour" comme s'il s'agissait d'une question juridique

---

### Première tentative de solution — Liste de mots-clés (approche naïve)

La première idée était de maintenir une liste statique de mots à bloquer :

```python
SALUTATIONS = {"bonjour", "salut", "hello", "bonsoir", ...}
HORS_SUJET_KEYWORDS = ["météo", "sport", "film", "merci", "ça va", ...]

def _est_hors_sujet(question: str) -> bool:
    if question.lower() in SALUTATIONS:
        return True
    for kw in HORS_SUJET_KEYWORDS:
        if kw in question.lower():
            return True
    return False
```

**Limites identifiées :**

- **Incomplète par nature** : le vocabulaire hors-sujet est infini. Il est impossible de lister tous les sujets non juridiques.
- **Fragile** : "Bonjour, quelle est la clause de résiliation ?" serait bloqué à cause du mot "Bonjour".
- **Pas multilingue** : "Hello", "Hi", "Hola" nécessitent chacun une entrée manuelle.
- **Ne comprend pas le sens** : "Kézako ce contrat ?" passerait le filtre car aucun mot-clé ne correspond, alors que c'est une question pertinente.
- **Non maintenable** : la liste grossit indéfiniment au fil du temps.

Cette approche a donc été **abandonnée**.

---

### Solution retenue — Le LLM comme classificateur

Au lieu d'une liste statique, on utilise le LLM lui-même pour classifier la question **avant** de lancer le pipeline RAG :

```python
def _est_question_juridique(question: str) -> bool:
    reponse = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Tu es un classificateur. Réponds uniquement par OUI ou NON.\n"
                "La question suivante est-elle une question juridique relative à un contrat, "
                "une clause, une loi ou un droit ? "
                "(salutations, remerciements, sujets hors droit = NON)\n\n"
                f"Question : {question}"
            )
        }],
        temperature=0.0,
        max_tokens=5,
    )
    return "OUI" in reponse.choices[0].message.content.upper()
```

**Fonctionnement :**

```
Question utilisateur
        ↓
LLM classificateur : "Est-ce une question juridique ?" → OUI / NON
        ↓                              ↓
   Pipeline RAG complet          Réponse polie
   (embedding + ChromaDB         sans RAG
    + génération LLM)
```

---

### Comparaison des deux approches

| Critère | Liste de mots-clés | LLM classificateur |
|---|---|---|
| Couverture | Limitée aux mots listés | Illimitée, comprend le sens |
| Langues supportées | Seulement celles listées | Toutes |
| Fautes d'orthographe | Non gérées | Gérées |
| Phrases ambiguës | Mal gérées | Bien gérées |
| Coût | Zéro appel API | 1 appel léger (max 5 tokens) |
| Maintenabilité | Liste à maintenir manuellement | Aucune maintenance |
| Précision | Faible | Élevée |

---

### Exemples de classification

| Question | Liste mots-clés | LLM classificateur |
|---|---|---|
| "Bonjour" | BLOQUÉ | BLOQUÉ |
| "Bonjour, la clause 4 est-elle valide ?" | BLOQUÉ (faux positif) | AUTORISÉ |
| "Kézako ce contrat ?" | AUTORISÉ (faux négatif) | AUTORISÉ |
| "Parle-moi de cuisine" | Dépend de la liste | BLOQUÉ |
| "What is the termination clause?" | Non géré | AUTORISÉ |
| "Merci beaucoup" | BLOQUÉ | BLOQUÉ |

---

### Justification du coût

L'appel classificateur est volontairement minimal :
- `temperature=0.0` : réponse déterministe, pas de créativité
- `max_tokens=5` : le modèle répond uniquement "OUI" ou "NON"
- Coût estimé : **< 0.001$** par classification

Ce coût est négligeable comparé au coût d'un appel RAG complet (embedding + génération), qu'on évite pour toutes les questions hors-sujet.

---

### Lien avec la littérature

Cette approche s'inscrit dans la logique de **Magesh et al. (Stanford, 2024)** qui soulignent que chaque composant du pipeline RAG peut introduire des erreurs. Utiliser le LLM comme gardien en entrée du pipeline est une forme de **guardrail** qui protège la qualité et la cohérence des réponses — une pratique recommandée pour les applications RAG en domaine à risque élevé comme le droit.
