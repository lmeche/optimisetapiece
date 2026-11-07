"""Étude d'optimisation de l'équerre en U sur le cahier des charges SAE 5.1 (phase 3).

Objectif : minimiser masse / effort tenu (kg/kN).
Contraintes (<= 0 = respectée) :
  c0 = 9 810 / effort tenu - 1          pas de plastification à 1 t
  c1 = 1 - facteur de flambement / 3    flambement
  c2 = traction vis / Ft,Rd - 1         vis M10 classe 8.8, Ft,Rd = 0,9 x 800 x 58 / 1,25 = 33,4 kN
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts.equerre_u import EquerreU
from gabarit.evaluation import evaluer
from gabarit.optimisation import Bornes

RACINE = Path(__file__).resolve().parents[2]
CDC = RACINE / "cahiers" / "sae51_equerre.json"
FT_RD_M10_88 = 0.9 * 800 * 58.0 / 1.25  # N
FLAMBEMENT_MIN = 3.0
TAILLE_EXPLORATION = 8.0  # mm ; contrainte ~3 % au-dessus du maillage de 6 mm, donc du côté de la sécurité
B_MAX = 155.0
X_BOUT = 350.0  # bout de la semelle du socle

AFFINAGE = [Bornes("z_dessus", 100, 200), Bornes("u_b", 0, 1), Bornes("u_bas", 0, 1),
            Bornes("u_dos", 0, 1), Bornes("u_bout", 0, 1)]


def proposer(trial) -> dict:
    cdc = CahierDesCharges.charger(CDC)
    t = trial.suggest_categorical("t", cdc.procede.epaisseurs_disponibles)
    ecart = trial.suggest_categorical("ecart_vis", [0.0, 20.0, 40.0, 60.0])  # 0 = une seule colonne en y = 0
    z_dessus = trial.suggest_float("z_dessus", 100, 200)
    u_b = trial.suggest_float("u_b", 0, 1)
    u_bas = trial.suggest_float("u_bas", 0, 1)
    u_dos = trial.suggest_float("u_dos", 0, 1)
    u_bout = trial.suggest_float("u_bout", 0, 1)
    return parametres_concept(t, ecart, z_dessus, u_b, u_bas, u_dos, u_bout)


def parametres_concept(t, ecart, z_dessus, u_b, u_bas, u_dos, u_bout) -> dict:
    """Variables normalisées -> paramètres physiques, construits pour rester géométriquement cohérents."""
    b_min = max(2 * ecart + 24, 44.0)  # 44 : ailes juste sous le bord de l'appui Ø50
    b = b_min + u_b * (B_MAX - t - b_min)
    z_bas = 40 + u_bas * (z_dessus - 40 - 40)
    aile_min = 0.7 * 8 * t
    h_dos = aile_min + u_dos * max(0.0, z_dessus - z_bas - aile_min)
    h_bout = aile_min + u_bout * max(0.0, h_dos - aile_min)
    return dict(t=t, b=round(b, 2), z_dessus=round(z_dessus, 2), z_bas=round(z_bas, 2),
                h_aile_dos=round(h_dos, 2), h_aile_bout=round(h_bout, 2), x_bout=X_BOUT,
                colonnes_vis=(-ecart, ecart) if ecart > 0 else (0.0,))


def evaluer_params(p: dict, taille: float = TAILLE_EXPLORATION) -> tuple[float, list[float], dict]:
    os.environ.setdefault("GABARIT_THREADS", "1")
    cdc = CahierDesCharges.charger(CDC)
    concept = EquerreU(**p)
    erreurs = concept.verifier(cdc)
    if erreurs:
        return 1e3, [10.0, 10.0, 10.0], {"erreurs": erreurs}
    r = evaluer(concept, cdc, taille=taille, flambement=False)
    c0 = 9810.0 / r.effort_tenu - 1
    c2 = r.traction_vis_max / FT_RD_M10_88 - 1
    c1 = 0.0
    facteur = math.nan
    if c0 <= 0 and c2 <= 0:  # le flambement ne se calcule que si la statique passe (gain de temps)
        r = evaluer(concept, cdc, taille=taille, flambement=True)
        facteur = r.facteur_flambement
        c1 = 1 - facteur / FLAMBEMENT_MIN if math.isfinite(facteur) else 1.0
    mesures = {"masse_kg": r.masse_kg, "effort_tenu_N": r.effort_tenu, "sigma_utile": r.sigma_max_utile,
               "fleche": r.fleche, "flambement": facteur, "traction_vis": r.traction_vis_max}
    return r.masse_par_kN, [c0, c1, c2], mesures
