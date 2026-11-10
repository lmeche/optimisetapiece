"""Phase 1 : validation de la chaîne Gmsh + CalculiX contre la RDM.

Cas : tube carré soudé 40 x 40 x 2 mm (ligne moyenne 38 mm), longueur 1000 mm, acier, encastré à x = 0.
  1. Flexion : force de 1000 N en bout -> flèche et contrainte à mi-longueur comparées à la RDM.
  2. Flambement : compression en bout -> charge critique comparée à Euler (poutre encastrée-libre).
Critère : écart < 3 % en statique, < 5 % en flambement.
Lancer :  python tests/valider_solveur.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gabarit.fem.calculix import Materiau, ModeleCoque, lancer  # noqa: E402
from gabarit.fem.maillage import tube_carre  # noqa: E402

L, a, t = 1000.0, 38.0, 2.0
acier = Materiau("S235", E=210000.0, nu=0.3)
G = acier.E / (2 * (1 + acier.nu))
I = 2 * t * a**3 / 3  # tube mince, ligne moyenne
TOL = 1e-6


def modele(force_bout: tuple[float, float, float]) -> ModeleCoque:
    noeuds, elements = tube_carre(L, a, n_long=50, n_cote=6)
    encastres = {n for n, c in noeuds.items() if abs(c[0]) < TOL}
    bout = [n for n, c in noeuds.items() if abs(c[0] - L) < TOL]
    forces = {n: tuple(f / len(bout) for f in force_bout) for n in bout}
    return ModeleCoque(noeuds, elements, t, acier, encastres, forces)


def ecart(calc: float, theo: float) -> float:
    return 100 * (calc - theo) / theo


def main() -> int:
    ok = True
    dossier = Path(tempfile.mkdtemp(prefix="gabarit_"))

    # 1. Flexion
    F = 1000.0
    res = lancer(modele((0.0, 0.0, -F)), dossier, "flexion")
    U = res.deplacements()
    bout = [n for n, c in res.coords.items() if abs(c[0] - L) < 1e-3]
    fleche = -float(np.mean([U[n][2] for n in bout]))
    fleche_theo = F * L**3 / (3 * acier.E * I) + F * L / (G * 2 * a * t)
    milieu = [n for n, c in res.coords.items() if abs(c[0] - L / 2) < 1e-3]
    S = res.champ("STRESS")
    sigma = max(abs(S[n][0]) for n in milieu)
    sigma_theo = F * (L / 2) * (a / 2 + t / 2) / I
    for nom, calc, theo, tol in [("Flèche en bout (mm)", fleche, fleche_theo, 3),
                                 ("Contrainte à mi-longueur (MPa)", sigma, sigma_theo, 3)]:
        e = ecart(calc, theo)
        ok &= abs(e) < tol
        print(f"{nom:34s} calcul {calc:10.3f}  RDM {theo:10.3f}  écart {e:+6.2f} %  {'OK' if abs(e) < tol else 'ÉCHEC'}")

    # 2. Flambement. Effort de 30 kN, proche de la charge critique : avec une force unitaire (facteur de
    # l'ordre de 40 000), le solveur de valeurs propres de CalculiX rend un facteur faux environ une fois sur
    # deux (constaté : 0,8 ; 0,9 ; 19 au lieu de 37 778) ; avec un facteur de l'ordre de 1, il est stable.
    effort = 30000.0
    res = lancer(modele((-effort, 0.0, 0.0)), dossier, "flambement", analyse="flambement", n_modes=2)
    pcr = res.facteurs_flambement[0] * effort
    pcr_theo = math.pi**2 * acier.E * I / (4 * L**2)
    e = ecart(pcr, pcr_theo)
    ok &= abs(e) < 5
    print(f"{'Charge critique de flambement (N)':34s} calcul {pcr:10.1f}  Euler {pcr_theo:9.1f}  écart {e:+6.2f} %  {'OK' if abs(e) < 5 else 'ÉCHEC'}")

    print("\nValidation", "RÉUSSIE" if ok else "ÉCHOUÉE", f"(fichiers dans {dossier})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
