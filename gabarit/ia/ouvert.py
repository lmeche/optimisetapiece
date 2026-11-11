"""IA Concepteur avec un modèle ouvert, par une API « compatible OpenAI » (chat/completions).

Fonctionne avec Ollama ou LM Studio en local (gratuit, rien ne sort du PC) et avec les services qui
hébergent des modèles ouverts (OpenRouter, Mistral, Together…). Même prompt, mêmes outils de contrôle et
mêmes règles d'enregistrement que le moteur Claude (`gabarit/ia/concepteur.py`).

Deux façons pour le modèle de soumettre un concept :
- par appel d'outil (verifier_concept / enregistrer_concept), si le modèle et le serveur le permettent ;
- sinon dans le texte, dans un bloc ```json : Gabarit le vérifie, l'enregistre s'il est valide et
  renvoie le rapport au modèle pour qu'il corrige ou passe au concept suivant.

Adresses usuelles : Ollama http://localhost:11434/v1, LM Studio http://localhost:1234/v1,
OpenRouter https://openrouter.ai/api/v1, Mistral https://api.mistral.ai/v1.
Avec Ollama, agrandir la fenêtre de contexte (le prompt fait environ 8 000 jetons), par exemple
avec la variable d'environnement OLLAMA_CONTEXT_LENGTH=32768 avant de lancer Ollama.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.ia.concepteur import OUTILS, RACINE, TOURS_MAX, Session, _executer_outil, _relatif, prompt_systeme

URL_OLLAMA = "http://localhost:11434/v1"
CONSIGNE_TEXTE = (
    "\n\n## Si tu ne peux pas appeler d'outil\nPour chiffrer une idée, écris un bloc ```json "
    "{\"predimensionner\": {...}} (mêmes champs que l'outil) : Gabarit te répond avec le calcul. Puis écris le "
    "concept complet, avec sa prédiction, dans un bloc ```json (un seul concept par message). Gabarit le vérifie, "
    "l'enregistre s'il passe les contrôles et te répond avec le rapport. Quand tous les concepts demandés sont "
    "enregistrés, termine par le bilan sans bloc json.")


def _outils_openai() -> list[dict]:
    return [{"type": "function", "function": {"name": o["name"], "description": o["description"],
                                              "parameters": o["input_schema"]}} for o in OUTILS]


def concevoir_ouvert(n_concepts: int = 3, url: str = URL_OLLAMA, modele: str = "qwen3:8b", cle: str = "",
                     chemin_cdc: str | Path = RACINE / "cahiers" / "sae51_equerre.json",
                     exemples: list[Path] | None = None, sortie: Path = RACINE / "concepts" / "ia",
                     dossier_journal: Path | None = None, consigne: str = "", calcul: bool = True,
                     outils: bool = True, delai: float = 600.0, tours_max: int = TOURS_MAX,
                     lecons: str | None = None, afficher=print) -> Session:
    cdc = CahierDesCharges.charger(chemin_cdc)
    exemples = exemples if exemples is not None else [RACINE / "concepts" / "sae51_L_V.json"]
    session = Session(dossier_journal or RACINE / "resultats" / "concepteur" / time.strftime("%Y%m%d_%H%M%S"))
    quoi = "un concept" if n_concepts == 1 else f"{n_concepts} concepts différents"
    demande = (f"Propose {quoi} pour ce cahier des charges, chacun vérifié puis enregistré."
               + (f"\n\nConsigne de l'utilisateur : {consigne}" if consigne else ""))
    messages = [{"role": "system", "content": prompt_systeme(cdc, exemples, lecons, savoir_compact=True) + CONSIGNE_TEXTE},
                {"role": "user", "content": demande}]
    session.journal(f"# IA Concepteur (modèle ouvert)\n\nServeur {url}, modèle {modele}.\n\n**Demande** : {demande}")
    entetes = {"Content-Type": "application/json", **({"Authorization": f"Bearer {cle}"} if cle else {})}

    illisibles = 0
    for _ in range(tours_max):
        corps = {"model": modele, "messages": messages}
        if outils:
            corps["tools"] = _outils_openai()
        try:
            donnees = _poster(f"{url.rstrip('/')}/chat/completions", corps, entetes, delai)
        except ErreurServeur as e:
            if outils and "tool" in e.texte.lower():
                outils = False  # le modèle ne gère pas les outils : on passe au mode texte
                session.mode_texte = True
                session.journal("Le serveur refuse les outils pour ce modèle : passage au mode texte.")
                continue
            raise
        usage = donnees.get("usage") or {}
        session.jetons["entree"] += usage.get("prompt_tokens", 0) or 0
        session.jetons["sortie"] += usage.get("completion_tokens", 0) or 0
        choix = donnees["choices"][0]
        message = choix["message"]
        texte = re.sub(r"<think>.*?</think>", "", message.get("content") or "", flags=re.S)  # raisonnement de Qwen, DeepSeek…
        appels = message.get("tool_calls") or []
        if texte.strip():
            afficher(texte)
            session.journal(texte)
        messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})

        if appels:
            for a in appels:
                nom = a["function"]["name"]
                args = a["function"].get("arguments") or "{}"
                try:
                    entree = args if isinstance(args, dict) else json.loads(args)
                except ValueError:  # arguments tronqués ou mal formés : le modèle reçoit une erreur et recommence
                    entree = None
                resultat, erreur = _executer_outil(nom, entree, cdc, sortie, session, calcul)
                afficher(f"\n[{nom}] {resultat[:300]}{'…' if len(resultat) > 300 else ''}\n")
                session.journal(f"**Résultat de {nom}**{' (erreur)' if erreur else ''} :\n```json\n{resultat}\n```")
                messages.append({"role": "tool", "tool_call_id": a.get("id", nom), "content": resultat})
            continue

        demande_calcul = _predimensionnement_dans_texte(texte)
        if demande_calcul is not None:  # mode texte : le modèle chiffre son idée avant d'écrire le concept
            resultat, _ = _executer_outil("predimensionner", demande_calcul, cdc, sortie, session, calcul)
            afficher(f"\n[predimensionner] {resultat[:300]}{'…' if len(resultat) > 300 else ''}\n")
            session.journal(f"**Résultat de predimensionner** :\n```json\n{resultat}\n```")
            messages.append({"role": "user", "content": f"Calcul de predimensionnement :\n```json\n{resultat}\n```"})
            continue
        concept = _concept_dans_texte(texte)
        if concept is None:
            if "```" in texte and illisibles < 2:  # un bloc de code mal formé : on redemande
                illisibles += 1
                messages.append({"role": "user", "content": "Bloc ```json illisible : renvoie le concept complet en JSON valide."})
                continue
            session.bilan = texte.strip()  # le modèle conclut
            session.arret = ("terminé" if len(session.concepts) >= n_concepts
                             else f"le modèle a conclu avec {len(session.concepts)} concept(s) sur {n_concepts}")
            break
        illisibles = 0
        # concept rendu dans le texte : enregistré s'il passe les contrôles, sinon le rapport repart au modèle
        resultat, erreur = _executer_outil("enregistrer_concept", {"concept": concept}, cdc, sortie, session, calcul)
        if erreur and calcul:
            resultat, _ = _executer_outil("verifier_concept", {"concept": concept}, cdc, sortie, session, calcul)
        afficher(f"\n[contrôle] {resultat[:300]}{'…' if len(resultat) > 300 else ''}\n")
        session.journal(f"**Contrôle du concept**{' (refusé)' if erreur else ' (enregistré)'} :\n```json\n{resultat}\n```")
        if erreur:
            suite = "Concept refusé, voici le rapport de contrôle : corrige-le et renvoie-le en entier."
        elif len(session.concepts) < n_concepts:
            suite = f"Concept enregistré ({len(session.concepts)}/{n_concepts}). Passe au suivant."
        else:
            suite = "Tous les concepts sont enregistrés : écris le bilan, sans bloc json."
        messages.append({"role": "user", "content": f"{suite}\n```json\n{resultat}\n```"})
    else:
        session.arret = f"limite de {tours_max} tours atteinte"
    session.journal(f"## Bilan\n\n{session.bilan or '(aucun)'}\n\nArrêt : {session.arret}. "
                    f"Concepts : {', '.join(_relatif(p) for p in session.concepts) or 'aucun'}. "
                    f"Jetons : {session.jetons}.")
    return session


class ErreurServeur(Exception):
    def __init__(self, code: int, texte: str):
        super().__init__(f"le serveur a répondu {code} : {texte[:500]}")
        self.code, self.texte = code, texte


def _poster(url: str, corps: dict, entetes: dict, delai: float) -> dict:
    """POST JSON avec la bibliothèque standard (proxy et NO_PROXY pris dans l'environnement)."""
    requete = urllib.request.Request(url, data=json.dumps(corps).encode(), headers=entetes, method="POST")
    try:
        with urllib.request.urlopen(requete, timeout=delai) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ErreurServeur(e.code, e.read().decode("utf-8", "replace")) from None


def _predimensionnement_dans_texte(texte: str) -> dict | None:
    """Dernier bloc ```json du texte de la forme {"predimensionner": {...}}, s'il n'y a pas de concept après."""
    for bloc in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", texte, flags=re.DOTALL)):
        try:
            d = json.loads(bloc)
        except ValueError:
            continue
        if isinstance(d, dict) and "toles" in d.get("concept", d):
            return None  # un concept vient après : c'est lui qui compte
        if isinstance(d, dict) and isinstance(d.get("predimensionner"), dict):
            return d["predimensionner"]
    return None


def _concept_dans_texte(texte: str) -> dict | None:
    """Dernier bloc ```json du texte qui ressemble à un concept."""
    for bloc in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", texte, flags=re.S)):
        try:
            d = json.loads(bloc)
        except ValueError:
            continue
        d = d.get("concept", d) if isinstance(d, dict) else d
        if isinstance(d, dict) and "toles" in d:
            return d
    return None

