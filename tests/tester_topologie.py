"""Tests de l'optimisation topologique (sans CalculiX) : zone évidable, poches tirées d'éléments retirés,
concept figé avec ses poches, explication d'un candidat.

Lancer :  python tests/tester_topologie.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from gabarit.analyse import topologie as topo  # noqa: E402
from gabarit.cahier_des_charges import CahierDesCharges  # noqa: E402
from gabarit.concepts import evidements as ev  # noqa: E402
from gabarit.concepts.format import ConceptJSON  # noqa: E402
from gabarit.ia.outils import controler_structure  # noqa: E402

CDC = CahierDesCharges.charger(RACINE / "cahiers" / "sae51_equerre.json")
VALEURS = json.loads((RACINE / "tests" / "donnees" / "sae51_L_V_optimum.json").read_text(encoding="utf-8"))["valeurs"]
resultats: list[tuple[str, bool, str]] = []


def verifier(nom: str, ok: bool, detail: str = "") -> None:
    resultats.append((nom, bool(ok), detail))


def main() -> int:
    c = ConceptJSON.charger(RACINE / "concepts" / "sae51_L_V.json", VALEURS, CDC)
    m0, _ = c.modele(CDC, 8.0)
    faces_el = dict(c.faces_elements)
    geo = c.geometrie()

    # 1. zone évidable : une seule face par élément, quadrilatère dans la zone permise
    evd = topo.zone_evidable(c, m0, faces_el)
    tous = [e for ids in evd.values() for e in ids]
    verifier("zone évidable : le dessus en fait partie", "L.dessus" in evd, ", ".join(f"{f} {len(i)}" for f, i in evd.items()))
    verifier("zone évidable : aucun élément de doublage", all(len(faces_el[e]) == 1 for e in tous))
    ctx = c.contexte_evidements()
    f = geo.face("L.dessus")
    rep = ev.Repere.de(f.points, f.normale())
    permis = Polygon(rep.vers_2d(f.points)).buffer(-ev.ligament_min(f.t)).difference(ev.zones_protegees(f, ctx))
    dedans = [permis.buffer(1e-6).contains(Polygon(rep.vers_2d(np.array([m0.noeuds[n] for n in m0.elements[e][:4]]))))
              for e in evd["L.dessus"]]
    verifier("zone évidable : éléments dans la zone permise (bandes, zones protégées)", all(dedans))

    # 2. poches tirées d'un bloc d'éléments retirés au milieu du dessus
    centres = {e: np.mean([m0.noeuds[n] for n in m0.elements[e][:4]], axis=0) for e in evd["L.dessus"]}
    bloc = {e for e, p in centres.items() if 120 <= p[0] <= 230 and abs(p[1]) <= 40}
    poches = topo.poches(c, m0, faces_el, bloc)
    polys = poches.get("L.dessus", [])
    verifier("poches : une poche pour un bloc d'un seul tenant", len(polys) == 1, f"{len(polys)} poche(s), bloc de {len(bloc)} éléments")
    verifier("poches : dans la zone permise, sans îlot, assez grandes",
             all(permis.buffer(1e-3).contains(p) and not p.interiors and p.area >= ev.AIRE_MIN for p in polys),
             ", ".join(f"{p.area / 100:.0f} cm²" for p in polys))
    r = ev.rayon_defaut(f.t)
    ouverte = all(p.buffer(-r + 0.2).buffer(r - 0.2).symmetric_difference(p).area < 0.02 * p.area for p in polys)
    verifier("poches : coins arrondis au rayon de fabrication (ouverture morphologique stable)", ouverte)
    vide = topo.poches(c, m0, faces_el, set(list(evd["L.dessus"])[:1]))
    verifier("poches : un élément isolé ne donne pas de poche (moins de 500 mm²)", not vide)

    # 3. concept figé avec ses poches : se construit, poches en « polygone », masse diminuée de l'aire découpée
    d = topo.concept_avec_poches(c.definition, c.valeurs, geo, poches)
    verifier("concept figé : structure valide", controler_structure(d) == [], "; ".join(controler_structure(d)[:2]))
    verifier("concept figé : chaque variable n'a plus qu'une valeur",
             all(s["type"] == "choix" and len(s["valeurs"]) == 1 for s in d["variables"].values()))
    allege = ConceptJSON(d, c.valeurs, CDC)
    erreurs = allege.verifier(CDC)
    verifier("concept figé : géométrie valide aux mêmes valeurs", erreurs == [], "; ".join(erreurs[:2]))
    rho = CDC.materiau.rho
    attendu = c.masse(rho) - sum(p.area for p in polys) * f.t * rho * 1e-9
    verifier("concept figé : masse = masse d'origine - aire découpée x t x rho (0,5 %)",
             abs(allege.masse(rho) / attendu - 1) < 0.005, f"{allege.masse(rho):.4f} / {attendu:.4f} kg")

    # 4. explication d'un candidat : rapport des kg/kN = rapport des masses x rapport des contraintes utiles
    ref = {"masse_kg": 4.0, "sigma_utile": 150.0, "kg_par_kN": 4.0 * 150.0 / (9.81 * 235.0),
           "point_critique": [350.0, 0.0, 190.0], "faces_critiques": ["L.dessus"]}
    cand = topo.Candidat(3, 0.15, d, {"L.dessus": 1}, 4500.0, masse_kg=3.8, sigma_utile=180.0,
                         point_critique=[56.0, -74.0, 190.0], faces_critiques=["L.dessus", "L.retour_dos_g"])
    cand.kg_par_kN = cand.masse_kg * cand.sigma_utile / (9.81 * 235.0)
    texte = topo.explication(ref, cand)
    verifier("explication : identité kg/kN = masse x contrainte utile",
             f"x{cand.kg_par_kN / ref['kg_par_kN']:.3f}" in texte and "x0.950" in texte and "x1.200" in texte, texte)
    verifier("explication : déplacement du point critique signalé", "déplacé" in texte and "L.retour_dos_g" in texte)
    verifier("explication : kg/kN identique à la mesure", math.isclose(0.95 * 1.2, cand.kg_par_kN / ref["kg_par_kN"]))

    for nom, ok, detail in resultats:
        print(f"{'OK   ' if ok else 'ÉCHEC'} {nom}" + (f"  ({detail})" if detail else ""))
    echecs = sum(not ok for _, ok, _ in resultats)
    print(f"\n{len(resultats) - echecs}/{len(resultats)} tests réussis")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
