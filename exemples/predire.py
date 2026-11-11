"""Tester une prédiction physique : l'écrire dans le carnet, PUIS calculer le concept, PUIS juger.

Lancer depuis la racine du dépôt :
    python exemples/predire.py apprentissage/predictions/lv_diag_u2.json

Le fichier d'essai décrit :
    {"concept": "concepts/sae51_L_V.json",
     "nom": "L + V, diagonale en U de 2 mm",                      # nom de la fiche
     "valeurs_depuis": "tests/donnees/sae51_L_V_optimum.json",     # valeurs de départ (synthese.json ou dict)
     "modifications": {"t_V": 2, "h_ret_diag": 20},                # ce qui change
     "prediction": {"kg_par_kN": 0.41, "mode_limite": "plastification", "principes": ["P3", "P4"],
                    "raisonnement": "..."},                         # format de gabarit/savoir/prediction.py
     "experience": {"parent": "F001", "modification": "...", "sens": "hausse", "lecon": "L007"},  # facultatif
     "source": "qui a fait la prédiction"}

La prédiction (et l'expérience) sont enregistrées dans le carnet avant le calcul. Le concept est ensuite
mesuré comme une fiche du carnet (compte rendu à 6 mm avec flambement, kg/kN de contrôle à 8 mm), puis Gabarit
rend son verdict. Le compte rendu va dans resultats/predictions/<essai>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from gabarit.apprentissage.boucle import CDC, TAILLE_FICHE, amorcer
from gabarit.apprentissage.carnet import CARNET_DEFAUT, Carnet
from gabarit.savoir import prediction as sp


def _chemin(x: str) -> Path:
    p = Path(x)
    return p if p.is_absolute() else RACINE / p


def main() -> int:
    p = argparse.ArgumentParser(description="Prédiction physique : enregistrer, calculer, juger")
    p.add_argument("essai", help="fichier JSON de l'essai")
    p.add_argument("--carnet", default=str(CARNET_DEFAUT), help="carnet d'apprentissage")
    p.add_argument("--cdc", default=str(CDC), help="cahier des charges")
    p.add_argument("--taille", type=float, default=TAILLE_FICHE, help="maille de la fiche (mm)")
    p.add_argument("--controle", type=float, default=8.0, help="maille de contrôle du kg/kN (mm, 0 pour aucune)")
    a = p.parse_args()

    essai = json.loads(Path(a.essai).read_text(encoding="utf-8"))
    concept = _chemin(essai["concept"])
    valeurs = {}
    if essai.get("valeurs_depuis"):
        d = json.loads(_chemin(essai["valeurs_depuis"]).read_text(encoding="utf-8"))
        valeurs.update(d.get("valeurs", d))
    valeurs.update(essai.get("valeurs", {}))
    valeurs.update(essai.get("modifications", {}))
    pred, erreurs = sp.lire({"prediction": essai.get("prediction")})
    if pred is None or "kg_par_kN" not in pred or "mode_limite" not in pred:
        print("Prédiction illisible : " + "; ".join(erreurs or ["champ « prediction » absent"]))
        return 1

    carnet = Carnet.charger(Path(a.carnet))
    x = essai.get("experience") or None
    e = None
    if x:
        e = carnet.ajouter_experience(x["parent"], x["modification"], x.get("grandeur", "kg_par_kN"),
                                      x.get("sens", "baisse"), x.get("lecon"), source=essai.get("source", "humain"))
    pr = carnet.ajouter_prediction(pred, concept, essai.get("nom", ""), source=essai.get("source", "humain"),
                                   experience=e.id if e else None, erreurs=erreurs)
    carnet.enregistrer()  # la prédiction est sur le disque avant le moindre calcul
    print(f"Enregistré avant le calcul : {pr.texte()}" + (f"\n{e.texte()}" if e else ""))

    t0 = time.time()
    dossier = RACINE / "resultats" / "predictions" / Path(a.essai).stem
    fiche = amorcer(carnet, concept, valeurs, a.cdc, taille=a.taille, convergence_taille=a.controle or None,
                    dossier=dossier, origine=essai.get("source", "humain"), parent=x["parent"] if x else None,
                    experience=e.id if e else None)
    if essai.get("nom"):
        fiche.nom = essai["nom"]
    carnet.conclure_prediction(pr.id, fiche.id)
    if e:
        carnet.conclure_experience(e.id, fiche.id)
    carnet.enregistrer()
    print(f"\nFiche : {fiche.resume()}\nPrédiction : {pr.texte()}" + (f"\nExpérience : {e.texte()}" if e else "")
          + f"\n\n{time.time() - t0:.0f} s ; compte rendu dans {dossier}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
