"""Étude d'optimisation générique d'un concept au format déclaratif (phase 4).

Les variables du concept deviennent des paramètres Optuna :
  epaisseur -> choix parmi les épaisseurs de l'atelier ; choix -> choix ; reel -> réel borné ;
  fraction -> réel dans [0, 1] ; retour -> interrupteur (avec / sans) + fraction de hauteur entre
  l'aile minimale de pliage de sa tôle et son maximum ; evidement -> interrupteur (avec / sans) + fraction
  de marge entre la bande minimale de sa tôle (max(2 t, 10 mm)) et son maximum.
Objectif et contraintes identiques à l'étude de l'équerre en U (masse / effort tenu ; plastification,
flambement >= 3, vis M10 8.8, géométrie valide).
"""
from __future__ import annotations

import json
import math
import os
from functools import partial
from pathlib import Path

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts.evidements import ligament_min
from gabarit.concepts.format import ConceptJSON, evaluer_expr, vé_et_aile
from gabarit.evaluation import evaluer
from gabarit.optimisation import Bornes

FT_RD_M10_88 = 0.9 * 800 * 58.0 / 1.25  # N
FLAMBEMENT_MIN = 3.0
TAILLE_EXPLORATION = 8.0


class EtudeConcept:
    def __init__(self, chemin_concept: str | Path, chemin_cdc: str | Path):
        self.chemin_concept = str(chemin_concept)
        self.chemin_cdc = str(chemin_cdc)
        self.definition = json.loads(Path(chemin_concept).read_text(encoding="utf-8"))
        self.cdc = CahierDesCharges.charger(chemin_cdc)
        self.evaluer = partial(evaluer_concept, self.chemin_concept, self.chemin_cdc)

    def proposer(self, trial) -> dict:
        return proposer_valeurs(self.definition, self.cdc, trial)

    def affinage(self) -> list[Bornes]:
        b = []
        for nom, s in self.definition["variables"].items():
            if s["type"] == "reel":
                b.append(Bornes(nom, s["min"], s["max"]))
            elif s["type"] == "fraction":
                b.append(Bornes(nom, 0.0, 1.0))
            elif s["type"] in ("retour", "evidement"):
                b.append(Bornes(f"{nom}_u", 0.0, 1.0))
        return b


def proposer_valeurs(definition: dict, cdc: CahierDesCharges, trial) -> dict:
    """Valeurs des variables du concept tirées par `trial` (un essai Optuna ou tout objet qui fournit
    suggest_categorical et suggest_float)."""
    v = {}
    retours = {}
    evidements = {}
    for nom, s in definition["variables"].items():
        ty = s["type"]
        if ty == "epaisseur":
            v[nom] = trial.suggest_categorical(nom, cdc.procede.epaisseurs_disponibles)
        elif ty == "choix":
            v[nom] = trial.suggest_categorical(nom, s["valeurs"])
        elif ty == "reel":
            v[nom] = trial.suggest_float(nom, s["min"], s["max"])
        elif ty == "fraction":
            v[nom] = trial.suggest_float(nom, 0.0, 1.0)
        elif ty == "retour":
            retours[nom] = (trial.suggest_categorical(f"{nom}_actif", [False, True]),
                            trial.suggest_float(f"{nom}_u", 0.0, 1.0))
        elif ty == "evidement":
            evidements[nom] = (trial.suggest_categorical(f"{nom}_actif", [False, True]),
                               trial.suggest_float(f"{nom}_u", 0.0, 1.0))
        else:
            raise ValueError(f"type de variable inconnu : {ty}")
    for nom, (actif, u) in retours.items():
        v[nom] = hauteur_retour(definition, cdc, nom, actif, u, v)
    for nom, (actif, u) in evidements.items():
        v[nom] = marge_evidement(definition, nom, actif, u, v)
    return v


def hauteur_retour(definition: dict, cdc: CahierDesCharges, nom: str, actif: bool, u: float,
                   valeurs: dict) -> float:
    """0 si le retour est absent, sinon entre l'aile minimale de pliage de sa tôle et son maximum."""
    if not actif:
        return 0.0
    s = definition["variables"][nom]
    tole = next(t for t in definition["toles"] if t["nom"] == s["tole"])
    t = evaluer_expr(tole["epaisseur"], valeurs)
    _, aile = vé_et_aile(t, cdc.procede.ves_disponibles)
    return round(aile + u * max(0.0, s["max"] - aile), 2)


def marge_evidement(definition: dict, nom: str, actif: bool, u: float, valeurs: dict) -> float:
    """0 si l'évidement est absent, sinon une marge entre la bande minimale de sa tôle et son maximum."""
    if not actif:
        return 0.0
    s = definition["variables"][nom]
    tole = next(t for t in definition["toles"] if t["nom"] == s["tole"])
    bas = ligament_min(evaluer_expr(tole["epaisseur"], valeurs))
    return round(bas + u * max(0.0, s["max"] - bas), 2)


def evaluer_concept(chemin_concept: str, chemin_cdc: str, valeurs: dict,
                    taille: float = TAILLE_EXPLORATION) -> tuple[float, list[float], dict]:
    os.environ.setdefault("GABARIT_THREADS", "1")
    cdc = CahierDesCharges.charger(chemin_cdc)
    concept = ConceptJSON.charger(chemin_concept, valeurs, cdc)
    erreurs = concept.verifier(cdc)
    if erreurs:
        return 1e3, [10.0, 10.0, 10.0], {"erreurs": erreurs}
    try:
        r = evaluer(concept, cdc, taille=taille, flambement=False)
    except RuntimeError as e:  # assemblage non connexe, maillage impossible...
        return 1e3, [10.0, 10.0, 10.0], {"erreurs": [str(e)[:300]]}
    c0 = 9810.0 / r.effort_tenu - 1
    c2 = r.traction_vis_max / FT_RD_M10_88 - 1
    c1, facteur = 0.0, math.nan
    if c0 <= 0 and c2 <= 0:
        r = evaluer(concept, cdc, taille=taille, flambement=True)
        facteur = r.facteur_flambement
        c1 = 1 - facteur / FLAMBEMENT_MIN if math.isfinite(facteur) else 1.0
    mesures = {"masse_kg": float(r.masse_kg), "effort_tenu_N": r.effort_tenu, "sigma_utile": r.sigma_max_utile,
               "fleche": r.fleche, "flambement": facteur, "traction_vis": float(r.traction_vis_max)}
    return float(r.masse_par_kN), [c0, c1, c2], mesures
