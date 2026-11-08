"""Phase 4 : l'IA Concepteur propose des concepts pour le cahier des charges SAE 5.1.

Prérequis : une clé API Anthropic dans la variable d'environnement ANTHROPIC_API_KEY
(PowerShell : $env:ANTHROPIC_API_KEY = "sk-ant-..." ; bash : export ANTHROPIC_API_KEY=sk-ant-...).

Lancer depuis la racine du dépôt :
    python exemples/concevoir.py [nombre de concepts] [--effort high] [--consigne "texte"] [--sans-calcul]
Avec un modèle ouvert (Ollama, LM Studio, OpenRouter, Mistral…), sans clé Anthropic :
    python exemples/concevoir.py 2 --moteur ouvert --url http://localhost:11434/v1 --modele qwen3:8b
(clé éventuelle du service dans la variable GABARIT_IA_CLE).
Les concepts validés sont écrits dans concepts/ia/, le journal (raisonnement résumé, appels d'outils,
bilan) dans resultats/concepteur/<date>/journal.md. Ensuite, pour chaque concept :
    python exemples/optimiser_concept.py concepts/ia/<concept>.json 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
import anthropic  # noqa: E402

from gabarit.ia.concepteur import EFFORT, MODELE, concevoir  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="IA Concepteur de Gabarit")
    p.add_argument("n", nargs="?", type=int, default=3, help="nombre de concepts à proposer")
    p.add_argument("--effort", default=EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--moteur", default="claude", choices=["claude", "ouvert"],
                   help="claude (API Anthropic) ou ouvert (API compatible OpenAI : Ollama, LM Studio, OpenRouter...)")
    p.add_argument("--modele", default=None, help=f"défaut : {MODELE} pour claude, qwen3:8b pour ouvert")
    p.add_argument("--url", default="http://localhost:11434/v1", help="adresse du serveur (moteur ouvert)")
    p.add_argument("--sans-outils", action="store_true", help="moteur ouvert : concept rendu en JSON dans le texte")
    p.add_argument("--consigne", default="", help="indication libre transmise à l'IA (idée de forme, contrainte...)")
    p.add_argument("--sans-calcul", action="store_true", help="contrôle géométrique seul, sans calcul rapide")
    p.add_argument("--cdc", default=str(RACINE / "cahiers" / "sae51_equerre.json"), help="cahier des charges (JSON)")
    a = p.parse_args()
    if a.moteur == "ouvert":
        return ouvert(a)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Variable ANTHROPIC_API_KEY absente : crée une clé sur https://platform.claude.com puis définis-la.")
        return 1
    try:
        s = concevoir(a.n, chemin_cdc=a.cdc, modele=a.modele or MODELE, effort=a.effort, consigne=a.consigne,
                      calcul=not a.sans_calcul)
    except anthropic.AuthenticationError:
        print("Clé API refusée : vérifie ANTHROPIC_API_KEY.")
        return 1
    except anthropic.PermissionDeniedError as e:
        print(f"Accès refusé pour cette clé : {e.message}")
        return 1
    except anthropic.BadRequestError as e:
        print(f"Requête refusée par l'API : {e.message}")
        return 1
    except anthropic.RateLimitError:
        print("Limite de débit atteinte malgré les nouvelles tentatives : réessaie dans quelques minutes.")
        return 1
    except anthropic.APIStatusError as e:
        print(f"Erreur de l'API ({e.status_code}) : {e.message}")
        return 1
    except anthropic.APIConnectionError:
        print("Connexion à l'API impossible : vérifie le réseau ou le proxy.")
        return 1
    return bilan(s)


def ouvert(a) -> int:
    import urllib.error

    from gabarit.ia.ouvert import ErreurServeur, concevoir_ouvert

    try:
        s = concevoir_ouvert(a.n, url=a.url, modele=a.modele or "qwen3:8b", cle=os.environ.get("GABARIT_IA_CLE", ""),
                             chemin_cdc=a.cdc, consigne=a.consigne, calcul=not a.sans_calcul,
                             outils=not a.sans_outils)
    except ErreurServeur as e:
        print(f"Erreur du serveur d'IA : {e}")
        return 1
    except TimeoutError:
        print("Le modèle met trop de temps à répondre : essaie un modèle plus petit.")
        return 1
    except urllib.error.URLError as e:
        print(f"Serveur injoignable à {a.url} ({e.reason}) : Ollama ou LM Studio est-il lancé ?")
        return 1
    return bilan(s)


def bilan(s) -> int:
    print(f"\n\nArrêt : {s.arret}")
    print(f"Concepts enregistrés : {[str(c) for c in s.concepts] or 'aucun'}")
    print(f"Journal : {s.dossier / 'journal.md'}")
    print(f"Jetons : {s.jetons}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
