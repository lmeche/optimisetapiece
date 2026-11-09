"""Validation des évidements : plaque trouée en traction, concentration de contrainte comparée à la théorie.

Cas : plaque 400 x 100 x 3 mm percée en son centre d'un trou Ø30 (d / W = 0,3), passée par toute la chaîne
de Gabarit (format de concept, évidement « cercle », maillage raffiné au bord du trou, coques S8R, CalculiX).
Bord x = 0 encastré, effort de traction F réparti sur le bord x = 400 (forces nodales cohérentes).
Référence : facteur de concentration de Howland pour une bande de largeur finie, sous la forme de
Peterson (Pilkey, Peterson's Stress Concentration Factors, 3e éd., graphique 4.1) :
    Kt,net = 3,00 - 3,13 (d/W) + 3,66 (d/W)² - 1,53 (d/W)³, rapporté à la section nette (W - d) t.
Contrôles : contrainte au bord du trou (écart < 3 %), masse (aire percée exacte, écart < 0,5 %), et la même
contrainte sur un maillage plus grossier puis plus fin (convergence).
Lancer :  python tests/valider_evidements.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gabarit.cahier_des_charges import CahierDesCharges  # noqa: E402
from gabarit.concepts import format as fmt  # noqa: E402
from gabarit.concepts.format import ConceptJSON  # noqa: E402
from gabarit.fem.calculix import Materiau, ModeleCoque, lancer  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
L, W, t, d = 400.0, 100.0, 3.0, 30.0
F = 6000.0  # N
TOL = 1e-6


def plaque() -> dict:
    return {
        "nom": "plaque trouée", "description": "validation",
        "variables": {"d": {"type": "reel", "min": 10, "max": 40}},
        "toles": [{"nom": "P", "epaisseur": t,
                   "faces": {"plaque": [[0, -W / 2, 100], [L, -W / 2, 100], [L, W / 2, 100], [0, W / 2, 100]]},
                   "evidements": [{"face": "plaque", "type": "cercle", "centre": [L / 2, 0, 100], "diametre": "d"}]}],
        "fixation": {"face": "P.plaque", "colonnes_vis": 0}, "appui": {"face": "P.plaque"}, "justifications": {},
    }


def kt_net_peterson(r: float) -> float:
    return 3.00 - 3.13 * r + 3.66 * r**2 - 1.53 * r**3


def contrainte_bord(taille: float, relative: float, dossier: Path) -> tuple[float, int]:
    fmt.TAILLE_TROU_RELATIVE = relative
    cdc = CahierDesCharges.charger(RACINE / "cahiers" / "sae51_equerre.json")
    c = ConceptJSON(plaque(), {"d": d}, cdc)
    noeuds, elements, epaisseurs = c.mailler(taille)
    encastres = {n for n, p in noeuds.items() if abs(p[0]) < TOL}
    forces: dict[int, np.ndarray] = {}
    q = F / W  # N/mm le long du bord chargé
    for conn in elements.values():  # S8R : sommets 0-3, milieux 4 (0-1), 5 (1-2), 6 (2-3), 7 (3-0)
        for a, b, m in ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)):
            if abs(noeuds[conn[a]][0] - L) < TOL and abs(noeuds[conn[b]][0] - L) < TOL:
                le = abs(noeuds[conn[a]][1] - noeuds[conn[b]][1])
                for n, part in ((conn[a], 1 / 6), (conn[b], 1 / 6), (conn[m], 4 / 6)):
                    forces[n] = forces.get(n, np.zeros(3)) + np.array([q * le * part, 0.0, 0.0])
    acier = Materiau("S235", E=210000.0, nu=0.3)
    modele = ModeleCoque(noeuds, elements, t, acier, encastres, {n: tuple(f) for n, f in forces.items()},
                         epaisseurs=epaisseurs)
    res = lancer(modele, dossier, f"plaque_{taille:g}_{relative:.3f}")
    vm = res.von_mises()
    bord = [n for n, p in res.coords.items() if abs(math.hypot(p[0] - L / 2, p[1]) - d / 2) < 0.05]
    return max(vm[n] for n in bord), len(elements)


def main() -> int:
    ok = True
    dossier = Path(tempfile.mkdtemp(prefix="gabarit_trou_"))
    sigma_net = F / ((W - d) * t)
    theo = kt_net_peterson(d / W) * sigma_net
    print(f"Plaque {L:g} x {W:g} x {t:g} mm, trou Ø{d:g}, traction {F:g} N : contrainte nette {sigma_net:.2f} MPa, "
          f"Kt,net (Peterson) {kt_net_peterson(d / W):.3f}, contrainte au bord attendue {theo:.1f} MPa\n")
    calcul = {}
    for taille, rel in ((8.0, 1 / 2), (8.0, fmt.TAILLE_TROU_RELATIVE), (8.0, 1 / 6), (4.0, 1 / 6)):
        s, n = contrainte_bord(taille, rel, dossier)
        calcul[(taille, rel)] = s
        e = 100 * (s - theo) / theo
        print(f"maille {taille:g} mm, bord du trou {rel * d / 2:4.1f} mm : {n:5d} éléments, contrainte {s:7.1f} MPa, "
              f"écart {e:+5.2f} %")
    defaut = calcul[(8.0, 1 / 3)]
    e = 100 * (defaut - theo) / theo
    ok &= abs(e) < 3
    print(f"\nRéglage retenu (maille au bord = rayon / 3) : écart {e:+.2f} %  {'OK' if abs(e) < 3 else 'ÉCHEC'}")
    fin = calcul[(4.0, 1 / 6)]
    conv = 100 * abs(defaut - fin) / fin
    ok &= conv < 3
    print(f"Écart au maillage le plus fin : {conv:.2f} %  {'OK' if conv < 3 else 'ÉCHEC'}")

    cdc = CahierDesCharges.charger(RACINE / "cahiers" / "sae51_equerre.json")
    m = ConceptJSON(plaque(), {"d": d}, cdc).masse(7850.0)
    m_theo = (L * W - math.pi * d**2 / 4) * t * 7850.0 * 1e-9
    em = 100 * (m - m_theo) / m_theo
    ok &= abs(em) < 0.5
    print(f"Masse : {m * 1000:.2f} g pour {m_theo * 1000:.2f} g attendus, écart {em:+.3f} %  "
          f"{'OK' if abs(em) < 0.5 else 'ÉCHEC'}")
    print("\nValidation", "RÉUSSIE" if ok else "ÉCHOUÉE", f"(fichiers dans {dossier})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
