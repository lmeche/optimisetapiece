"""Exploration longue : un modèle open source propose des concepts toute la nuit, une piste par essai.

Prérequis : un serveur de modèle compatible OpenAI lancé sur le PC (LM Studio ou Ollama) avec une fenêtre de
contexte d'au moins 16 000 jetons, ou un service hébergé (OpenRouter, Mistral) avec sa clé dans GABARIT_IA_CLE.

Lancer depuis la racine du dépôt, par exemple pour 8 heures avec LM Studio :
    python exemples/explorer.py --heures 8 --url http://localhost:1234/v1 --modele qwen3-8b
Les concepts valides arrivent dans concepts/exploration/, le catalogue de tous les essais (piste, résultat,
masse, effort tenu, kg/kN) dans resultats/exploration/<date>/catalogue.csv. Ctrl+C arrête proprement : le
catalogue est écrit après chaque essai.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from gabarit.ia.exploration import explorer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Exploration longue de formes avec un modèle open source")
    p.add_argument("--heures", type=float, default=8.0, help="durée de l'exploration (défaut 8 h)")
    p.add_argument("--essais", type=int, default=None, help="nombre d'essais maximum (défaut : pas de limite)")
    p.add_argument("--url", default="http://localhost:11434/v1", help="adresse du serveur (Ollama, LM Studio...)")
    p.add_argument("--modele", default="qwen3:8b", help="nom du modèle sur ce serveur")
    p.add_argument("--sans-outils", action="store_true", help="concept rendu en JSON dans le texte")
    p.add_argument("--tours", type=int, default=6, help="tours de correction par essai (défaut 6)")
    p.add_argument("--delai", type=float, default=1800, help="attente maximale d'une réponse, en secondes")
    p.add_argument("--sans-calcul", action="store_true", help="contrôle géométrique seul, sans calcul rapide")
    p.add_argument("--cdc", default=str(RACINE / "cahiers" / "sae51_equerre.json"), help="cahier des charges (JSON)")
    a = p.parse_args()
    try:
        explorer(heures=a.heures, essais_max=a.essais, url=a.url, modele=a.modele,
                 cle=os.environ.get("GABARIT_IA_CLE", ""), chemin_cdc=a.cdc, outils=not a.sans_outils,
                 tours=a.tours, calcul=not a.sans_calcul, delai=a.delai)
    except KeyboardInterrupt:
        print("\nExploration interrompue : le catalogue contient les essais terminés.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
