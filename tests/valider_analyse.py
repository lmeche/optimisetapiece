"""Validation de l'analyse détaillée (gabarit/analyse) sur des cas à solution connue, puis sur le concept L + V.

1. Console en flexion (plaque 100 x 20 x 2 encastrée, effort au bout en forces nodales cohérentes) :
   - énergie des éléments = travail de l'effort (théorème de Clapeyron), à 1e-4 près ;
   - part de l'énergie en flexion > 99 % loin de l'encastrement ;
   - contrainte de peau aux points de Gauss = théorie des poutres 6 F (L - x) / (b t²), à 1 % près ;
   - L_eq = 3 L² / (2 t) (fibre neutre : la moyenne de |sigma| dans l'épaisseur vaut la moitié de la peau).
2. Tirant (même plaque en traction) : flexion nulle, contrainte moyenne F / A, L_eq = L.
3. Colonne comprimée (même plaque, effort de compression) : premier facteur de flambement proche d'Euler
   pi² E I / (4 L²) et déformée du mode 1 qui croît de l'encastrement vers le bout.
4. Garde-fou du solveur : bilan d'énergie et d'équilibre exact quand des efforts tombent sur des noeuds bloqués,
   solution fausse détectée (celle du solveur multi-tâche : énergie +349 %, réactions décalées de 24 %),
   mécanisme refusé après relances.
5. Concept L + V (valeurs optimales figées dans tests/donnees) : identité kg/kN = 1000 rho/Re x L_eq / eta
   exacte, énergie = travail, parts de masse et de L_eq qui se somment, faces du mode de flambement.

Lancer :  python tests/valider_analyse.py
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from gabarit.analyse.champ import champ_elements  # noqa: E402
from gabarit.analyse.compte_rendu import analyser  # noqa: E402
from gabarit.cahier_des_charges import CahierDesCharges  # noqa: E402
from gabarit.concepts.format import ConceptJSON  # noqa: E402
from gabarit.fem.calculix import (  # noqa: E402
    Materiau,
    ModeleCoque,
    bilan_correct,
    bilan_statique,
    lancer,
)

L, B, T, E = 100.0, 20.0, 2.0, 210000.0
resultats: list[tuple[str, bool, str]] = []


def verifier(nom: str, ok: bool, detail: str = "") -> None:
    resultats.append((nom, bool(ok), detail))


def plaque(effort: tuple[float, float, float], nx: int = 20, ny: int = 4) -> ModeleCoque:
    """Plaque S8R dans le plan z = 0, encastrée en x = 0, effort total `effort` réparti en forces nodales
    cohérentes (1/6, 4/6, 1/6 par bord d'élément) sur le bord x = L."""
    noeuds, elements = {}, {}

    def nid(i, j):
        return 1 + i + j * (2 * nx + 1)
    for j in range(2 * ny + 1):
        for i in range(2 * nx + 1):
            if not (i % 2 and j % 2):
                noeuds[nid(i, j)] = (L * i / (2 * nx), -B / 2 + B * j / (2 * ny), 0.0)
    k = 1
    for ej in range(ny):
        for ei in range(nx):
            i, j = 2 * ei, 2 * ej
            elements[k] = [nid(i, j), nid(i + 2, j), nid(i + 2, j + 2), nid(i, j + 2),
                           nid(i + 1, j), nid(i + 2, j + 1), nid(i + 1, j + 2), nid(i, j + 1)]
            k += 1
    forces: dict[int, np.ndarray] = {}
    for c in elements.values():
        for a, b, m in ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)):
            if abs(noeuds[c[a]][0] - L) < 1e-9 and abs(noeuds[c[b]][0] - L) < 1e-9:
                le = abs(noeuds[c[a]][1] - noeuds[c[b]][1]) / B
                for n, part in ((c[a], 1 / 6), (c[b], 1 / 6), (c[m], 4 / 6)):
                    forces[n] = forces.get(n, np.zeros(3)) + np.array(effort) * le * part
    encastres = {n for n, p in noeuds.items() if p[0] == 0}
    return ModeleCoque(noeuds, elements, T, Materiau("acier", E, 0.3), encastres,
                       {n: tuple(f) for n, f in forces.items()})


def travail(m: ModeleCoque, res) -> float:
    return 0.5 * sum(float(np.dot(f, res.u_noeuds[n])) for n, f in m.forces.items())


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="gabarit_analyse_"))

    # 1. console en flexion
    F = 10.0
    m = plaque((0.0, 0.0, F))
    res = lancer(m, dossier, "flexion", details=True)
    c = champ_elements(m, {e: ["P"] for e in m.elements}, res)
    W = travail(m, res)
    verifier("flexion : énergie des éléments = travail de l'effort", abs(c.energie.sum() / W - 1) < 1e-4,
             f"{c.energie.sum():.5f} / {W:.5f} N.mm")
    loin = (c.centre[:, 0] > 20) & (c.centre[:, 0] < 80)
    verifier("flexion : part de l'énergie en flexion > 99 %", c.part_flexion[loin].min() > 0.99,
             f"min {c.part_flexion[loin].min():.4f}")
    e = int(np.argmin(np.abs(c.centre[:, 0] - 52.5) + np.abs(c.centre[:, 1] - 1.25)))
    xg = c.centre[e, 0] - 2.5 / math.sqrt(3)  # point de Gauss amont de l'élément (le plus chargé)
    theo = 6 * F * (L - xg) / (B * T**2)
    verifier("flexion : peau = théorie des poutres (1 %)", abs(c.vm_peau[e] / theo - 1) < 0.01,
             f"{c.vm_peau[e]:.2f} / {theo:.2f} MPa")
    l_eq = float((c.volume * c.vm_moyen).sum() / F)
    verifier("flexion : L_eq = 3 L² / (2 t) (3 %)", abs(l_eq / (3 * L**2 / (2 * T)) - 1) < 0.03,
             f"{l_eq:.0f} / {3 * L**2 / (2 * T):.0f} mm")

    # 2. tirant
    F = 1000.0
    m = plaque((F, 0.0, 0.0))
    res = lancer(m, dossier, "traction", details=True)
    c = champ_elements(m, {e: ["P"] for e in m.elements}, res)
    W = travail(m, res)
    verifier("traction : énergie des éléments = travail de l'effort", abs(c.energie.sum() / W - 1) < 1e-4,
             f"{c.energie.sum():.5f} / {W:.5f} N.mm")
    verifier("traction : aucune flexion", c.part_flexion.max() < 1e-3, f"max {c.part_flexion.max():.2e}")
    verifier("traction : contrainte moyenne = F / A (0,5 %)", abs(c.vm_moyen[loin].mean() / (F / (B * T)) - 1) < 0.005,
             f"{c.vm_moyen[loin].mean():.3f} / {F / (B * T):.3f} MPa")
    l_eq = float((c.volume * c.vm_moyen).sum() / F)
    verifier("traction : L_eq = L (1 %)", abs(l_eq / L - 1) < 0.01, f"{l_eq:.1f} / {L:.0f} mm")

    # 3. colonne comprimée : facteur et déformée du mode 1
    m = plaque((-10.0, 0.0, 0.0))
    rb = lancer(m, dossier, "colonne", analyse="flambement", n_modes=3, fichiers=False, details=True)
    euler = math.pi**2 * E * (B * T**3 / 12) / (4 * L**2) / 10.0
    verifier("colonne : facteur du mode 1 proche d'Euler (5 %)", abs(rb.facteurs_flambement[0] / euler - 1) < 0.05,
             f"{rb.facteurs_flambement[0]:.2f} / {euler:.2f}")
    mode = rb.modes[0]
    flec = [np.mean([abs(mode[n][2]) for n, p in m.noeuds.items() if abs(p[0] - x) < 1e-9]) for x in (25, 50, 75, 100)]
    verifier("colonne : la déformée du mode 1 croît vers le bout", all(a < b for a, b in itertools.pairwise(flec)) and
             len(rb.modes) == len(rb.facteurs_flambement), " < ".join(f"{v:.3f}" for v in flec))

    # 4. garde-fou du solveur : bilan d'énergie et d'équilibre de chaque calcul statique
    m = plaque((0.0, 0.0, 10.0))
    n0 = min(m.encastres)
    m.forces[n0] = (50.0, 20.0, 30.0)  # effort sur un noeud encastré : il passe dans l'appui sans travailler
    milieu = sorted(n for n, p in m.noeuds.items() if abs(p[0] - L / 2) < 1e-9)
    m.appuis_x = set(milieu)
    m.forces[milieu[0]] = (40.0, 0.0, 5.0)  # effort sur un noeud bloqué en X seulement
    r = lancer(m, dossier, "garde")
    verifier("garde-fou : bilan exact avec efforts sur des noeuds bloqués", r.essais == 1 and bilan_correct(r.bilan)
             and abs(r.bilan["energie"]) < 1e-4 and r.bilan["equilibre"] < 1e-5,
             f"énergie {r.bilan['energie']:+.1e}, équilibre {r.bilan['equilibre']:.1e}")
    faux = copy.copy(r)
    faux.energie = {e: 4.49 * v for e, v in r.energie.items()}  # le calcul faux observé : énergie +349 %
    faux.reactions = {"ENCASTRE": {n: 0.76 * v for n, v in r.reactions["ENCASTRE"].items()}, "APPUI": r.reactions["APPUI"]}
    b = bilan_statique(m, faux)
    verifier("garde-fou : solution fausse détectée", not bilan_correct(b),
             f"énergie {b['energie']:+.0%}, équilibre {b['equilibre']:.2f}")
    m.encastres, m.appuis_x = set(), set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lancer(m, dossier, "mecanisme")
        verifier("garde-fou : mécanisme refusé", False, "aucune erreur levée")
    except RuntimeError as e:
        verifier("garde-fou : mécanisme refusé", "fausse" in str(e), str(e).split(" (")[0])

    # 5. concept L + V
    cdc = CahierDesCharges.charger(RACINE / "cahiers" / "sae51_equerre.json")
    valeurs = json.loads((RACINE / "tests" / "donnees" / "sae51_L_V_optimum.json").read_text(encoding="utf-8"))["valeurs"]
    concept = ConceptJSON.charger(RACINE / "concepts" / "sae51_L_V.json", valeurs, cdc)
    cr = analyser(concept, cdc, taille=8.0, flambement=True, dossier=dossier / "LV")
    k = cr.mesures["kg_par_kN"]
    verifier("L + V : kg/kN = 1000 rho/Re x L_eq / eta (identité exacte)",
             abs(cr.controle["identite_kg_par_kN"] / k - 1) < 1e-9, f"{cr.controle['identite_kg_par_kN']:.6f} / {k:.6f}")
    verifier("L + V : énergie des éléments = travail de l'effort (1 %)", abs(cr.controle["ecart_energie_pct"]) < 1.0,
             f"{cr.controle['ecart_energie_pct']:+.3f} %")
    verifier("L + V : parts de masse et de L_eq des faces sommées = 1",
             abs(sum(f.part_masse for f in cr.faces) - 1) < 1e-9
             and abs(sum(f.l_eq_mm for f in cr.faces) / cr.l_eq_mm - 1) < 1e-9,
             f"{sum(f.part_masse for f in cr.faces):.6f}, {sum(f.l_eq_mm for f in cr.faces):.1f} / {cr.l_eq_mm:.1f} mm")
    verifier("L + V : masse maillée = masse de la géométrie (0,1 %)",
             abs(cr.controle["masse_maillee_kg"] / cr.mesures["masse_kg"] - 1) < 1e-3,
             f"{cr.controle['masse_maillee_kg']:.4f} / {cr.mesures['masse_kg']:.4f} kg")
    verifier("L + V : faces du mode de flambement identifiées", bool(cr.flambement.get("faces")),
             ", ".join(f"{f} {p:.2f}" for f, p in cr.flambement.get("faces", [])[:3]))
    verifier("L + V : une carte par face", all(f.carte for f in cr.faces), f"{len(cr.faces)} faces")

    for nom, ok, detail in resultats:
        print(f"{'OK   ' if ok else 'ÉCHEC'} {nom}" + (f"  ({detail})" if detail else ""))
    echecs = sum(not ok for _, ok, _ in resultats)
    print(f"\n{len(resultats) - echecs}/{len(resultats)} contrôles réussis (fichiers dans {dossier})")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
