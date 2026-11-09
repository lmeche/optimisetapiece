"""Export STEP d'un concept : chaque face devient une plaque de son épaisseur (plis vifs, sans rayon), percée
de ses évidements.

Les faces d'une même tôle sont fusionnées en un solide ; chaque tôle reste un corps séparé (pièces à
souder). Une face doublée contre une face d'une autre tôle (même plan) est décalée de la demi-somme des
épaisseurs, du côté opposé au socle, pour que les deux tôles ne se chevauchent pas dans la maquette.
"""
from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np

from gabarit.concepts.format import ConceptJSON, surface_percee


def _decalages(concept: ConceptJSON) -> dict[str, np.ndarray]:
    """Décalage du plan moyen de chaque face doublée (la face de fixation reste contre le socle)."""
    g = concept.geometrie()
    dec = {}
    for i, a in enumerate(g.faces):
        for b in g.faces[i + 1:]:
            if a.tole == b.tole:
                continue
            na, nb = a.normale(), b.normale()
            if abs(abs(np.dot(na, nb)) - 1) > 1e-6 or abs(np.dot(b.points[0] - a.points[0], na)) > 1e-6:
                continue
            if not _recouvrement(a, b):
                continue
            fixe, mobile = (a, b) if b.id != g.fixation else (b, a)
            n = fixe.normale()
            if abs(n[0]) > 0.5:
                n = n * np.sign(n[0])  # vers +x : loin du socle
            elif n[2] < 0:
                n = -n
            dec[mobile.id] = n * (fixe.t + mobile.t) / 2
    return dec


def _recouvrement(a, b) -> bool:
    axes = np.linalg.svd(a.points - a.points.mean(0))[2][:2]
    pa, pb = (a.points - a.points[0]) @ axes.T, (b.points - a.points[0]) @ axes.T
    lo = np.maximum(pa.min(0), pb.min(0))
    hi = np.minimum(pa.max(0), pb.max(0))
    return bool((hi - lo > 1.0).all())


def exporter_step(concept: ConceptJSON, chemin: str | Path) -> Path:
    chemin = Path(chemin)
    g = concept.geometrie()
    dec = _decalages(concept)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("gabarit_step")
    occ = gmsh.model.occ
    par_tole: dict[str, list[tuple[int, int]]] = {}
    for f in g.faces:
        n = f.normale()
        decalage = -n * f.t / 2 + dec.get(f.id, 0)
        s = surface_percee(occ, f, points=f.points + decalage, decalage=decalage)  # avec ses évidements
        vol = [e for e in occ.extrude([(2, s)], *(n * f.t)) if e[0] == 3]
        par_tole.setdefault(f.tole, []).extend(vol)
    for vols in par_tole.values():
        if len(vols) > 1:
            try:
                occ.fuse(vols[:1], vols[1:])
            except Exception:  # fusion refusée par OCC : les plaques restent des solides séparés
                pass
    occ.synchronize()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(chemin))
    gmsh.finalize()
    return chemin
