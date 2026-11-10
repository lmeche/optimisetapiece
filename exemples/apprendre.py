"""Apprentissage (version 2) : l'IA conçoit, Gabarit optimise et mesure, l'IA analyse, le carnet retient.

Chaque cycle teste une expérience prévue, améliore le meilleur concept ou explore un principe nouveau (voir
gabarit/apprentissage/boucle.py). Le carnet (apprentissage/carnet.json, et carnet.md lisible) garde les fiches
des concepts mesurés, les leçons et leurs preuves, les expériences et leurs verdicts ; il nourrit le prompt de
l'IA Concepteur au cycle suivant.

Lancer depuis la racine du dépôt :
    python exemples/apprendre.py --cycles 3                         # Claude conçoit et analyse (ANTHROPIC_API_KEY)
    python exemples/apprendre.py --cycles 20 --moteur ouvert --url http://localhost:1234/v1 --modele qwen3-8b
    python exemples/apprendre.py --cycles 20 --moteur ouvert --analyste claude   # formes gratuites, analyse par Claude
    python exemples/apprendre.py --afficher                         # le carnet, lisible
    python exemples/apprendre.py --analyser F001 F002               # l'IA Analyste relit des fiches, sans calcul
    python exemples/apprendre.py --amorcer concepts/x.json --valeurs resultats/x/synthese.json
                                                                    # fiche d'un concept optimisé à part
Modèle ouvert hébergé (OpenRouter, Mistral) : clé dans la variable GABARIT_IA_CLE. Ctrl+C arrête : le carnet est
enregistré après chaque étape.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from gabarit.apprentissage.boucle import (  # noqa: E402
    BUDGET_ENFANT,
    BUDGET_EXPLORATION,
    CDC,
    URL_OLLAMA,
    Reglages,
    amorcer,
    analyser_ia,
    apprendre,
)
from gabarit.apprentissage.carnet import CARNET_DEFAUT, Carnet  # noqa: E402


def _budget(texte: str) -> tuple[int, int, int]:
    morceaux = tuple(int(x) for x in texte.split(","))
    if len(morceaux) != 3 or min(morceaux) < 0:
        raise argparse.ArgumentTypeError("budget attendu : initial,bayésien,affinage (trois entiers)")
    return morceaux


def main() -> int:
    p = argparse.ArgumentParser(description="Boucle d'apprentissage de Gabarit")
    p.add_argument("--cycles", type=int, default=1, help="nombre de cycles (défaut 1)")
    p.add_argument("--moteur", default="claude", choices=["claude", "ouvert"], help="IA Concepteur")
    p.add_argument("--modele", default="", help="défaut : claude-opus-5-5 (claude), qwen3:8b (ouvert)")
    p.add_argument("--analyste", default="", choices=["", "claude", "ouvert", "aucun"],
                   help="IA Analyste (défaut : le moteur du concepteur)")
    p.add_argument("--modele-analyste", default="", help="modèle de l'IA Analyste")
    p.add_argument("--url", default=URL_OLLAMA, help="adresse du serveur du modèle ouvert (Ollama, LM Studio...)")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"], help="Claude")
    p.add_argument("--sans-outils", action="store_true", help="modèle ouvert : concept rendu en JSON dans le texte")
    p.add_argument("--delai", type=float, default=1800, help="modèle ouvert : attente maximale d'une réponse (s)")
    p.add_argument("--budget", type=_budget, default=BUDGET_ENFANT,
                   help="essais d'optimisation d'un enfant : initial,bayésien,affinage (défaut %(default)s)")
    p.add_argument("--budget-exploration", type=_budget, default=BUDGET_EXPLORATION,
                   help="essais d'optimisation d'un concept nouveau (défaut %(default)s)")
    p.add_argument("--paralleles", type=int, default=4, help="calculs en parallèle pendant l'optimisation")
    p.add_argument("--explorer-tous", type=int, default=3, help="un cycle d'exploration tous les N cycles (0 : jamais)")
    p.add_argument("--carnet", default=str(CARNET_DEFAUT), help="fichier du carnet (JSON)")
    p.add_argument("--cdc", default=str(CDC), help="cahier des charges (JSON)")
    p.add_argument("--afficher", action="store_true", help="afficher le carnet et s'arrêter")
    p.add_argument("--analyser", nargs="*", metavar="FICHE", help="analyser ces fiches (toutes si aucune) et s'arrêter")
    p.add_argument("--amorcer", metavar="CONCEPT", help="ajouter la fiche d'un concept déjà optimisé et s'arrêter")
    p.add_argument("--valeurs", help="avec --amorcer : JSON des valeurs (synthese.json d'optimiser_concept.py ou dict)")
    a = p.parse_args()

    carnet = Carnet.charger(Path(a.carnet))
    reglages = Reglages(moteur=a.moteur, modele=a.modele, analyste=a.analyste, modele_analyste=a.modele_analyste,
                        url=a.url, cle=os.environ.get("GABARIT_IA_CLE", ""), effort=a.effort,
                        chemin_cdc=Path(a.cdc), budget=a.budget, budget_exploration=a.budget_exploration,
                        n_paralleles=a.paralleles, explorer_tous=a.explorer_tous, outils=not a.sans_outils,
                        delai=a.delai)
    if a.afficher:
        print(carnet.texte())
        return 0
    if a.amorcer:
        if not a.valeurs:
            p.error("--amorcer demande --valeurs")
        d = json.loads(Path(a.valeurs).read_text(encoding="utf-8"))
        valeurs = d.get("valeurs", d)
        f = amorcer(carnet, Path(a.amorcer).resolve(), valeurs, reglages.chemin_cdc,
                    optimisation={"essais": d.get("essais"), "faisables": d.get("faisables"), "taille": 8.0}
                    if "essais" in d else None)
        carnet.enregistrer()
        print(f.resume())
        return 0
    moteurs = {reglages.moteur_analyste()} if a.analyser is not None else {reglages.moteur, reglages.moteur_analyste()}
    if "claude" in moteurs and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Il faut une clé API Anthropic dans la variable d'environnement ANTHROPIC_API_KEY "
              "(ou --moteur ouvert / --analyste ouvert pour un modèle ouvert).")
        return 1
    if a.analyser is not None:
        ids = a.analyser or [f.id for f in carnet.fiches]
        inconnues = [i for i in ids if not carnet.a_fiche(i)]
        if inconnues:
            p.error(f"fiches inconnues : {', '.join(inconnues)}")
        if reglages.moteur_analyste() == "aucun":
            p.error("--analyser demande une IA Analyste (claude ou ouvert)")
        bilan = analyser_ia(reglages, carnet, ids)
        carnet.enregistrer()
        print(f"\nCarnet enregistré : {carnet.chemin}" + (f" (jetons {bilan.jetons})" if bilan else ""))
        return 0
    try:
        apprendre(a.cycles, reglages, carnet)
    except KeyboardInterrupt:
        carnet.enregistrer()
        print("\nApprentissage interrompu : le carnet garde les étapes terminées.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
