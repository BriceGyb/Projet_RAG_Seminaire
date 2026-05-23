import os
import re
import pypdf
from typing import List


def load_pdf_text(pdf_path: str) -> str:
    """Charge le texte brut d'un fichier PDF page par page."""
    texte = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            texte += page.extract_text() or ""
    return texte


def clean_text(text: str) -> str:
    """Nettoyage basique : normalisation des espaces et suppression des blancs."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Découpage RCTS (Recursive Character Text Splitter).
    Tente les séparateurs dans l'ordre : paragraphes, lignes, phrases, mots.
    Ajoute un overlap entre chunks consécutifs pour conserver le contexte.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: List[str]) -> List[str]:
        if not text.strip():
            return []
        if len(text) <= chunk_size:
            return [text.strip()]

        sep = seps[0]
        remaining = seps[1:]

        # Découpage forcé caractère par caractère en dernier recours
        if sep == "":
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    sous_chunks = _split(part, remaining)
                    if sous_chunks:
                        chunks.extend(sous_chunks[:-1])
                        current = sous_chunks[-1]
                    else:
                        current = ""
                else:
                    current = part

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if c]

    raw_chunks = _split(text, separators)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    # Injection de l'overlap : on préfixe chaque chunk (sauf le premier)
    # avec la queue du chunk précédent
    final_chunks = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        queue_precedente = raw_chunks[i - 1][-overlap:]
        final_chunks.append((queue_precedente + " " + raw_chunks[i]).strip())

    return final_chunks


def load_pdf_pages(pdf_path: str) -> List[dict]:
    """Charge le texte d'un PDF page par page, en conservant les numéros de page."""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": page_num, "text": text})
    return pages


# Détecte une ligne commençant par un numéro d'article suivi d'un point : "1385. ..." ou "3.1. ..."
_ARTICLE_RE = re.compile(r'^(\d{1,4}(?:\.\d+)?)\.\s+\S')


def extract_articles_from_pdf(pdf_path: str) -> List[dict]:
    """
    Extrait les articles numérotés d'un PDF juridique.
    Retourne une liste de dicts : {law, article_id, page, text}.
    """
    law_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = load_pdf_pages(pdf_path)

    articles = []
    current = None

    for page_data in pages:
        page_num = page_data["page"]
        for line in page_data["text"].split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            match = _ARTICLE_RE.match(stripped)
            if match:
                if current:
                    current["text"] = re.sub(r"\s+", " ", current["text"]).strip()
                    if current["text"]:
                        articles.append(current)

                article_id = match.group(1)
                # texte après "1385. "
                rest = stripped[len(article_id) + 2:].strip()
                current = {
                    "law": law_name,
                    "article_id": article_id,
                    "page": page_num,
                    "text": rest,
                }
            elif current is not None:
                current["text"] += " " + stripped

    if current:
        current["text"] = re.sub(r"\s+", " ", current["text"]).strip()
        if current["text"]:
            articles.append(current)

    return articles


if __name__ == "__main__":
    texte_test = "Ceci est un paragraphe.\n\nVoici un second paragraphe.\n\nEt un troisième."
    chunks = chunk_text(texte_test, chunk_size=50, overlap=10)
    print(f"{len(chunks)} chunks produits :")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {repr(c)}")
