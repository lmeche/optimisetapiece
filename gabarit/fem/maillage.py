"""Maillage coque (quadrangles 8 noeuds) avec Gmsh à partir de surfaces moyennes."""
from __future__ import annotations

import gmsh


def _extraire_quads8() -> tuple[dict, dict]:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    noeuds = {int(t): tuple(coords[3 * k : 3 * k + 3]) for k, t in enumerate(tags)}
    etags, enoeuds = gmsh.model.mesh.getElementsByType(16)  # 16 = quadrangle 8 noeuds
    elements = {int(e): [int(n) for n in enoeuds[8 * k : 8 * k + 8]] for k, e in enumerate(etags)}
    utilises = {n for conn in elements.values() for n in conn}
    noeuds = {n: c for n, c in noeuds.items() if n in utilises}
    return noeuds, elements


def tube_carre(L: float, a: float, n_long: int = 50, n_cote: int = 6):
    """Tube carré le long de X, côté a mesuré sur la ligne moyenne des parois."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tube")
    occ = gmsh.model.occ
    h = a / 2
    coins = [(-h, -h), (h, -h), (h, h), (-h, h)]
    p0 = [occ.addPoint(0, y, z) for y, z in coins]
    p1 = [occ.addPoint(L, y, z) for y, z in coins]
    long_ = [occ.addLine(p0[i], p1[i]) for i in range(4)]
    tr0 = [occ.addLine(p0[i], p0[(i + 1) % 4]) for i in range(4)]
    tr1 = [occ.addLine(p1[i], p1[(i + 1) % 4]) for i in range(4)]
    faces = []
    for i in range(4):
        boucle = occ.addCurveLoop([tr0[i], long_[(i + 1) % 4], -tr1[i], -long_[i]])
        faces.append(occ.addPlaneSurface([boucle]))
    occ.synchronize()
    for c in long_:
        gmsh.model.mesh.setTransfiniteCurve(c, n_long + 1)
    for c in tr0 + tr1:
        gmsh.model.mesh.setTransfiniteCurve(c, n_cote + 1)
    for f in faces:
        gmsh.model.mesh.setTransfiniteSurface(f)
        gmsh.model.mesh.setRecombine(2, f)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    noeuds, elements = _extraire_quads8()
    gmsh.finalize()
    return noeuds, elements
