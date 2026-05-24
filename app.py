import tempfile
import os
import streamlit as st

# Injection de la clé API depuis st.secrets (Streamlit Cloud) ou depuis data/.env (local)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from utils import load_pdf_text, clean_text
from analyze_contract import extraire_clauses, analyser_clause, repondre_question

st.set_page_config(
    page_title="Assistant Juridique",
    page_icon="⚖️",
    layout="wide",
)

IMPORTANCE_COLOR = {"haute": "🔴", "moyenne": "🟡", "faible": "🟢"}


def _est_question_juridique(question: str) -> bool:
    """Utilise le LLM comme filtre : détermine si la question mérite un appel RAG."""
    from config import client, CHAT_MODEL
    reponse = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Tu es un classificateur. Réponds uniquement par OUI ou NON.\n"
                "La question suivante est-elle une question juridique relative à un contrat, "
                "une clause, une loi ou un droit ? (salutations, remerciements, sujets hors droit = NON)\n\n"
                f"Question : {question}"
            )
        }],
        temperature=0.0,
        max_tokens=5,
    )
    return "OUI" in reponse.choices[0].message.content.upper()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ Assistant Juridique")
    st.caption("Analyse de contrats · Droit québécois & canadien")
    st.divider()

    uploaded_file = st.file_uploader("Uploader un contrat (PDF)", type="pdf")

    if uploaded_file:
        if st.button("Analyser le contrat", type="primary", use_container_width=True):
            # Sauvegarde temporaire du PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Réinitialise l'état
            st.session_state.pop("structure", None)
            st.session_state.pop("analyses", None)
            st.session_state.pop("messages", None)

            with st.spinner("Extraction du texte..."):
                texte = clean_text(load_pdf_text(tmp_path))
            os.unlink(tmp_path)

            if not texte.strip():
                st.error("Impossible d'extraire le texte de ce PDF.")
                st.stop()

            with st.spinner("Identification des clauses..."):
                structure = extraire_clauses(texte)
            st.session_state["structure"] = structure

            clauses = structure.get("clauses", [])
            analyses = []
            progress = st.progress(0, text="Analyse juridique des clauses...")
            for i, clause in enumerate(clauses):
                progress.progress((i + 1) / len(clauses),
                                  text=f"Clause {i+1}/{len(clauses)} : {clause['titre']}")
                analyses.append(analyser_clause(clause))
            progress.empty()

            st.session_state["analyses"] = analyses
            st.session_state["messages"] = []
            st.success(f"{len(analyses)} clauses analysées.")

    # Infos contrat si disponible
    if "structure" in st.session_state:
        s = st.session_state["structure"]
        st.divider()
        st.markdown(f"**Type :** {s.get('type_contrat', '—')}")
        parties = s.get("parties", [])
        if parties:
            st.markdown("**Parties :**")
            for p in parties:
                st.markdown(f"- {p}")


# ── Contenu principal ─────────────────────────────────────────────────────────

if "analyses" not in st.session_state:
    st.markdown("## Bienvenue")
    st.markdown(
        "Uploadez un contrat PDF dans la barre latérale pour démarrer l'analyse juridique.\n\n"
        "L'assistant identifiera les clauses importantes, les croisera avec le corpus de lois "
        "(Code civil du Québec, lois fédérales…) et répondra à vos questions."
    )
    st.stop()

structure = st.session_state["structure"]
analyses = st.session_state["analyses"]

tab_analyse, tab_qa = st.tabs(["📋 Analyse des clauses", "💬 Questions & Réponses"])


# ── Onglet 1 : Analyse ────────────────────────────────────────────────────────

with tab_analyse:
    st.subheader(f"Contrat · {structure.get('type_contrat', '').capitalize()}")

    hautes   = [a for a in analyses if a["importance"] == "haute"]
    moyennes = [a for a in analyses if a["importance"] == "moyenne"]
    faibles  = [a for a in analyses if a["importance"] == "faible"]

    for groupe, label in [(hautes, "Importance haute"), (moyennes, "Importance moyenne"), (faibles, "Importance faible")]:
        if not groupe:
            continue
        icone = IMPORTANCE_COLOR.get(groupe[0]["importance"], "⚪")
        st.markdown(f"### {icone} {label}")
        for a in groupe:
            with st.expander(f"**{a['titre']}**"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Texte de la clause**")
                    st.info(a["texte_original"])
                    st.markdown("**Analyse juridique**")
                    st.markdown(a["analyse"])
                with col2:
                    st.markdown("**Sources légales**")
                    for src in a["sources_legales"]:
                        st.markdown(f"- `{src}`")


# ── Onglet 2 : Q&R ────────────────────────────────────────────────────────────

with tab_qa:
    st.subheader("Questions sur le contrat")
    st.caption("Posez vos questions sur les clauses, leur validité ou les risques.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Historique
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Saisie
    question = st.chat_input("Votre question juridique sur ce contrat...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not _est_question_juridique(question):
                reponse = "Je suis un assistant juridique spécialisé dans l'analyse de contrats. Je ne peux répondre qu'aux questions relatives au contrat chargé et aux lois applicables."
                st.markdown(reponse)
            else:
                with st.spinner("Recherche juridique..."):
                    reponse = repondre_question(question, structure, analyses)
                st.markdown(reponse)

        st.session_state["messages"].append({"role": "assistant", "content": reponse})
