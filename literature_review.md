# Revue de Littérature — RAG Juridique
## Liens entre les articles scientifiques et la réalisation du projet

---

## Article 1 — Legal Chunking: Evaluating Methods for Effective Legal Text Retrieval

**Auteurs :** Andrea Filippo Ferraris, Davide Audrito, Giovanni Siragusa (informaticiens) — Alessandro Piovano (juriste)
**Institution :** Université de Bologne / Université de Turin
**Conférence :** JURIX 2024 — 37e International Conference on Legal Knowledge and Information Systems (IOS Press, doi:10.3233/FAIA241255)

---

### Ce que fait l'article

L'article compare trois méthodes de chunking appliquées au RGPD (Règlement Général sur la Protection des Données) dans un pipeline RAG, en mesurant la similarité cosinus entre les chunks récupérés et 17 questions juridiques de référence.

**Les trois méthodes testées :**

| Méthode | Principe | Score moyen |
|---|---|---|
| Simple Text Splitting | Découpe tous les N tokens peu importe le contenu | 0.41 |
| Recursive Text Splitting (RCTS) | Découpe aux frontières naturelles `\n\n`, `\n`, `. ` dans l'ordre | 0.41 |
| Semantic Chunking | Un modèle NLP détecte les ruptures sémantiques automatiquement | 0.33 |

**Résultat principal :** Aucune méthode ne produit de scores élevés. Le Semantic Chunking est le pire malgré son coût computationnel élevé. Le RCTS est légèrement supérieur car il respecte la structure du texte.

**Conclusion des auteurs :** Les textes juridiques nécessitent des approches qui respectent leur **structure hiérarchique** (articles, alinéas, sections) — ce qu'aucune des trois méthodes testées ne fait complètement.

---

### Leçons tirées pour notre projet

**Leçon 1 — Choix du RCTS comme méthode de chunking de base**

L'article confirme que le RCTS est la meilleure méthode parmi les approches automatiques testées. Notre implémentation dans `utils.py` suit exactement ce principe :

```python
separators = ["\n\n", "\n", ". ", " ", ""]
```

On tente de couper aux paragraphes d'abord, puis aux lignes, puis aux phrases, puis aux mots — exactement le Recursive Text Splitting de l'article.

**Leçon 2 — Aller au-delà du RCTS : extraction par articles**

La conclusion de l'article recommande de respecter la hiérarchie juridique. On a implémenté cette recommandation avec `extract_articles_from_pdf()` dans `utils.py` : chaque article de loi (ex: "Art. 1385") devient une unité sémantique indépendante dans ChromaDB, avec ses métadonnées (`article_id`, `page`, `law`). Notre approche va donc **au-delà** de ce que l'article teste.

**Leçon 3 — Ne pas utiliser le Semantic Chunking**

L'article démontre que le Semantic Chunking performe le pire sur les textes juridiques à cause des clauses imbriquées et des références croisées entre articles. On a écarté cette approche pour notre projet.

---

## Article 2 — LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain

**Auteurs :** Nicholas Pipitone, Ghita Houir Alami
**Institution :** ZeroEntropy, San Francisco
**Publication :** arXiv:2408.10343, août 2024

---

### Ce que fait l'article

Pipitone & Alami créent le **premier benchmark dédié à évaluer la partie retrieval** des pipelines RAG juridiques. Avant cet article, les benchmarks existants (LegalBench) évaluaient seulement la génération du LLM — personne n'évaluait la qualité du retrieval.

**Le dataset LegalBench-RAG :**
- 6 858 paires question-réponse annotées manuellement par des juristes
- 4 corpus de contrats réels (NDA, contrats commerciaux, fusions-acquisitions, politiques de confidentialité)
- Chaque réponse pointe vers un **span exact** dans le document (indices de caractères précis)
- Corpus total : 79 millions de caractères sur 714 documents

**Les 4 configurations testées :**

| Configuration | Precision@1 | Recall@64 |
|---|---|---|
| Naive seul (chunks 500 chars) | 2.40% | 76.39% |
| Naive + Reranker Cohere | 6.41% | 62.22% |
| **RCTS seul** | **6.41%** | **62.22%** |
| RCTS + Reranker Cohere | 6.13% | 61.06% |

**Résultat principal :** RCTS sans reranker = meilleure configuration. Le reranker généraliste (Cohere) **dégrade** les résultats sur les textes juridiques spécialisés car il n'est pas entraîné sur ce domaine.

**Point important :** L'article utilise exactement le même modèle d'embedding que notre projet : `text-embedding-3-large` d'OpenAI.

---

### Leçons tirées pour notre projet

**Leçon 1 — Validation scientifique du choix du RCTS**

L'article confirme expérimentalement, sur un large corpus de contrats réels, que le RCTS est la meilleure méthode de chunking pour le juridique. Notre implémentation est directement validée par ces résultats.

**Leçon 2 — Ne pas implémenter de reranker**

L'article démontre qu'un reranker généraliste nuit aux performances sur les textes juridiques. On a donc volontairement écarté cette étape post-retrieval de notre pipeline. Ce choix est scientifiquement justifié.

**Leçon 3 — Validation du modèle d'embedding**

L'article utilise `text-embedding-3-large` d'OpenAI et obtient ses meilleurs résultats avec ce modèle. Notre `config.py` utilise exactement ce modèle :

```python
EMBEDDING_MODEL = "text-embedding-3-large"
```

**Leçon 4 — Importance de la précision du retrieval**

L'article insiste sur le fait que récupérer des chunks **précis et minimaux** est préférable à de larges blocs de texte. Les chunks trop larges dépassent la fenêtre de contexte du LLM et augmentent les hallucinations. C'est pour ça qu'on sous-découpe les articles trop longs (`MAX_ARTICLE_LENGTH = 1500` dans `ingest.py`).

---

## Article 3 — Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools

**Auteurs :** Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, Daniel E. Ho
**Institution :** Stanford University / Yale University
**Publication :** Stanford Law, Journal of Empirical Legal Studies, 2024

---

### Ce que fait l'article

L'article conduit la **première évaluation empirique pré-enregistrée** des outils RAG juridiques commerciaux. Les auteurs testent 202 questions juridiques réelles sur LexisNexis (Lexis+ AI), Thomson Reuters (Ask Practical Law AI) et GPT-4 (sans RAG, comme référence).

Ces deux outils commerciaux prétendaient publiquement être "hallucination-free" grâce au RAG.

**Résultats :**

| Système | Réponses correctes et citées | Hallucinations | Incomplètes |
|---|---|---|---|
| **Lexis+ AI** | 65% | **17%** | 18% |
| **Thomson Reuters** | 18% | **17%** | 62% |
| **GPT-4 (sans RAG)** | ~35% | **58-82%** | — |

**L'article introduit une typologie des hallucinations en deux dimensions :**
- **Incorrectness** : la réponse contient une information factuellement fausse
- **Misgrounded** : la réponse est vraie mais cite une mauvaise source (potentiellement plus dangereux car difficile à détecter)

**Conclusion principale :** Le RAG **réduit** les hallucinations (de 58-82% à 17%) mais ne les **élimine pas**. La principale cause est un mauvais retrieval : si le mauvais document est récupéré, le LLM hallucine même avec RAG.

---

### Leçons tirées pour notre projet

**Leçon 1 — Contraindre le modèle au contexte récupéré**

L'article prouve que laisser le LLM répondre librement même avec RAG produit encore 17% d'hallucinations. On a donc imposé des contraintes strictes dans nos prompts :

```python
# rag.py et analyze_contract.py
"Répondez uniquement à partir du contexte fourni"
"Si la réponse ne peut pas être déterminée à partir du contexte, indiquez-le explicitement"
temperature = 0.1  # quasi-déterministe, pas de créativité
```

**Leçon 2 — Toujours afficher la source précise**

L'article introduit la notion de *misgrounded* : une réponse vraie mais citant une mauvaise source est une hallucination dangereuse car subtile. Pour contrer ça, notre interface Streamlit affiche systématiquement la source exacte de chaque information :

```
Sources : CCQ-1991.pdf art.1385 p.42
```

L'utilisateur peut vérifier lui-même — c'est exactement ce que Stanford recommande pour la supervision humaine des outils d'IA juridique.

**Leçon 3 — Investir sur la qualité du retrieval**

L'article démontre que la cause principale d'hallucination est un **retrieval de mauvaise qualité**. Si le mauvais chunk est récupéré, le LLM hallucine. C'est pourquoi on a investi sur :
- Le modèle d'embedding le plus performant (`text-embedding-3-large`)
- Le chunking par articles juridiques (unités sémantiques naturelles)
- L'overlap de 120 caractères entre chunks pour préserver le contexte

**Leçon 4 — Responsabilité du superviseur humain**

L'article souligne que les avocats doivent superviser et vérifier les outputs de l'IA. Notre système affiche toujours les sources et les chunks utilisés, ce qui permet à l'utilisateur (avocat, étudiant en droit) de vérifier la réponse avant de l'utiliser.

---

## Synthèse — Ce que la littérature a apporté au projet

| Décision de conception | Article source | Implémentation dans le projet |
|---|---|---|
| Utiliser RCTS comme méthode de chunking | Ferraris et al. + Pipitone & Alami | `chunk_text()` dans `utils.py` |
| Extraire les articles juridiques comme unités | Ferraris et al. (recommandation future work) | `extract_articles_from_pdf()` dans `utils.py` |
| Ne pas utiliser le Semantic Chunking | Ferraris et al. | Absent du projet |
| Ne pas utiliser de reranker | Pipitone & Alami | Absent du pipeline |
| Utiliser `text-embedding-3-large` | Pipitone & Alami | `config.py` |
| Prompt strict ancré dans le contexte | Magesh et al. | `rag.py`, `analyze_contract.py` |
| `temperature = 0.1` | Magesh et al. | `rag.py`, `analyze_contract.py` |
| Afficher la source précise (art. + page) | Magesh et al. | Interface Streamlit + métadonnées ChromaDB |
