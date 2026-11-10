"""Optimisation topologique (ESO) d'un concept à ses valeurs optimisées : carte de la matière la moins utile, puis
poches fabricables tirées de cette carte et recalculées par la chaîne complète (voir gabarit/analyse/topologie.py).

Lancer depuis la racine du dépôt :
    python exemples/alleger.py concepts/sae51_L_V.json --valeurs tests/donnees/sae51_L_V_optimum.json
    python exemples/alleger.py concepts/ia/x.json --valeurs resultats/x/synthese.json --taille 6
Sortie dans resultats/alleger/<concept>/ : topologie.md (itérations, candidats, cartes), candidat_eso_<part>.json
pour chaque niveau recalculé et concept_allege.json si un candidat faisable bat le concept sans poche. Un candidat
entre au carnet d'apprentissage par :
    python exemples/apprendre.py --amorcer resultats/alleger/<concept>/candidat_eso_15.json --valeurs <mêmes valeurs>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from gabarit.analyse.topologie import NIVEAUX, enregistrer, optimiser_topologie
from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts.format import ConceptJSON


def main() -> int:
    p = argparse.ArgumentParser(description="Optimisation topologique (ESO) d'un concept")
    p.add_argument("concept", help="concept au format JSON")
    p.add_argument("--valeurs", required=True,
                   help="JSON des valeurs des variables (synthese.json d'optimiser_concept.py, ou dict)")
    p.add_argument("--taille", type=float, default=6.0, help="maille (mm, défaut 6)")
    p.add_argument("--niveaux", default=",".join(f"{100 * n:g}" for n in NIVEAUX),
                   help="parts retirées (en %%) dont on tire des poches à recalculer (défaut %(default)s)")
    p.add_argument("--cdc", default=str(RACINE / "cahiers" / "sae51_equerre.json"), help="cahier des charges")
    p.add_argument("--sortie", default="", help="dossier de sortie (défaut resultats/alleger/<concept>)")
    a = p.parse_args()

    d = json.loads(Path(a.valeurs).read_text(encoding="utf-8"))
    valeurs = d.get("valeurs", d)
    niveaux = tuple(float(x) / 100 for x in a.niveaux.split(",") if x.strip())
    cdc = CahierDesCharges.charger(a.cdc)
    concept = ConceptJSON.charger(a.concept, valeurs, cdc)
    erreurs = concept.verifier(cdc)
    if erreurs:
        print("Concept invalide à ces valeurs : " + "; ".join(erreurs))
        return 1
    dossier = Path(a.sortie) if a.sortie else RACINE / "resultats" / "alleger" / Path(a.concept).stem
    t0 = time.time()
    res = optimiser_topologie(concept, cdc, taille=a.taille, niveaux=niveaux, dossier=dossier / "calculs")
    retenu = enregistrer(res, dossier)
    print(res.texte())
    print(f"\n{time.time() - t0:.0f} s ; résultats dans {dossier}"
          + (f" ; concept allégé retenu : {retenu}" if retenu else " ; aucun candidat retenu"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
