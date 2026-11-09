"""Évidements : découpes d'allègement dans les tôles (géométrie, zones protégées, règles de fabrication).

Un évidement se déclare dans une tôle, comme un retour, en nommant la face qu'il perce :

    "evidements": [
      {"face": "dos", "type": "decalage", "marge": "m_dos"},
      {"face": "dos", "type": "decalage", "marge": "m_dos", "rayon": 15,
       "nervures": [[[0, 0.5], [2, 0.5]]], "largeur_nervure": "w_dos"},
      {"face": "diag", "type": "cercle", "centre": ["x_t", 0, "z_t"], "diametre": "d_t"},
      {"face": "dessus", "type": "polygone", "points": [[x, y, z], ...], "rayon": 12}
    ]

- « decalage » : poche qui suit le contour de la face à la distance « marge » (largeur de la bande de tôle
  laissée sur les bords), moins les zones protégées, coins arrondis au « rayon ». Une marge de 0 = pas de poche.
  Une nervure relie deux points du bord, chacun donné par [arête, position] (arête i : du point i au point
  i + 1 de la face, position de 0 à 1) : elle partage la poche en deux, comme la barre d'un treillis.
- « cercle » : trou de « diametre » (0 = pas de trou) centré en « centre », point projeté sur la face.
- « polygone » : découpe aux sommets « points » (projetés sur la face), coins arrondis au « rayon ».
Chaque évidement accepte « si » : il n'existe que si l'expression est > 0. Toutes les valeurs sont des
nombres ou des expressions des variables, comme les points des faces.

Zones protégées, où l'on ne découpe pas :
- le long des plis de la tôle : 2,5 t + rayon intérieur (0,16 x vé) de chaque côté de la ligne de pli, pour
  que le trou ne se déforme pas au pliage ;
- le long des soudures (arête commune, jonction en T, doublage) : max(10 mm, 3 t) de la ligne de soudure ;
- autour des vis utilisées : 12 mm du centre du trou, comme pour les bords de la face de fixation ;
- sous l'appui : disque de rayon 35 mm (appui Ø50 et 10 mm de bord).
Règles : toute bande de matière (marge, nervure, entre deux trous, entre un trou et le bord) >= max(2 t, 10 mm) ;
rayon des coins >= max(t, 5 mm) ; une poche de moins de 500 mm² est ignorée (gain négligeable) ; une zone
protégée qui se retrouverait isolée au milieu d'une poche est reliée au bord le plus proche par un bras de tôle
(aucun îlot ne flotte).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

TOL_PLAN = 1e-3  # mm : un point est dans le plan d'une face à cette distance près
AIRE_MIN = 500.0  # mm² : poche plus petite ignorée
MARGE_VIS = 12.0  # mm, du centre d'un trou de vis au bord d'une découpe (même règle que les bords du dos)
RAYON_APPUI_PROTEGE = 35.0  # mm : appui Ø50 + 10 mm de bord
SEGMENTS_QUART = 16  # segments par quart de cercle pour les arrondis


def ligament_min(t: float) -> float:
    """Largeur minimale d'une bande de tôle laissée par une découpe (bord, nervure, entre deux trous)."""
    return max(2.0 * t, 10.0)


def rayon_min(t: float) -> float:
    return max(t, 5.0)


def rayon_defaut(t: float) -> float:
    return max(2.0 * t, 10.0)


def marge_soudure(t: float) -> float:
    return max(10.0, 3.0 * t)


@dataclass
class Repere:
    """Repère plan d'une face : origine au premier point, e1 le long de la première arête."""

    o: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray

    @classmethod
    def de(cls, points: np.ndarray, normale: np.ndarray) -> Repere:
        o = points[0]
        e1 = points[1] - points[0]
        e1 = e1 - np.dot(e1, normale) * normale
        e1 = e1 / np.linalg.norm(e1)
        return cls(o, e1, np.cross(normale, e1), normale)

    def vers_2d(self, p) -> np.ndarray:
        q = np.asarray(p, float) - self.o
        return np.stack([q @ self.e1, q @ self.e2], axis=-1)

    def vers_3d(self, q) -> np.ndarray:
        q = np.asarray(q, float)
        return self.o + q[..., :1] * self.e1 + q[..., 1:2] * self.e2

    def distance_plan(self, p) -> np.ndarray:
        return (np.asarray(p, float) - self.o) @ self.n


@dataclass
class Contexte:
    """Ce qu'il faut savoir de l'assemblage pour protéger les zones d'une face."""

    faces: list  # toutes les faces (objets Face de format.py)
    fixation: str
    appui: str
    trous_vis: list[tuple[float, float]]  # (y, z) des vis utilisées
    point_charge: np.ndarray  # centre de l'appui (x, y, z)
    d_pli: Callable[[float], float]  # distance protégée de part et d'autre d'un pli, selon l'épaisseur


@dataclass
class ResultatEvidements:
    trous: list[np.ndarray] = field(default_factory=list)  # contours 3D (k, 3), sans répéter le premier point
    trous_2d: list[Polygon] = field(default_factory=list)
    rayons: list[float] = field(default_factory=list)  # rayon de courbure minimal de chaque contour (maillage)
    erreurs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def aire(self) -> float:
        return float(sum(p.area for p in self.trous_2d))


def zones_protegees(face, ctx: Contexte) -> BaseGeometry:
    """Union des zones de la face où l'on ne découpe pas, dans le repère plan de la face."""
    rep = Repere.de(face.points, face.normale())
    poly = Polygon(rep.vers_2d(face.points))
    zones: list[BaseGeometry] = []
    for g in ctx.faces:
        if g is face:
            continue
        ng = g.normale()
        paralleles = abs(abs(float(np.dot(rep.n, ng))) - 1) < 1e-9
        coplanaires = paralleles and abs(rep.distance_plan(g.points[0])) < TOL_PLAN
        if g.tole == face.tole:
            if coplanaires:  # même tôle, même plan : tôle continue découpée en deux faces, pas un pli
                continue
            for seg in _segments_communs(face, g):
                zones.append(_tampon(LineString(rep.vers_2d(seg)), ctx.d_pli(face.t)))
            continue
        ms = marge_soudure(max(face.t, g.t))
        if coplanaires:  # doublage : toute la zone recouverte, plus la marge de soudure
            inter = poly.intersection(Polygon(rep.vers_2d(g.points)))
            if inter.area > 1.0:
                zones.append(_tampon(inter, ms))
            continue
        # arêtes de g posées dans le plan de la face (jonction en T, arête commune)
        d = rep.distance_plan(g.points)
        for i in range(len(g.points)):
            j = (i + 1) % len(g.points)
            if abs(d[i]) < TOL_PLAN and abs(d[j]) < TOL_PLAN:
                ligne = LineString(rep.vers_2d(g.points[[i, j]])).intersection(poly.buffer(0.5))
                if ligne.length > 1.0:
                    zones.append(_tampon(ligne, ms))
        # arêtes de la face posées sur g (la face est l'âme d'un T soudé sur g)
        rg = Repere.de(g.points, ng)
        pg = Polygon(rg.vers_2d(g.points))
        dg = rg.distance_plan(face.points)
        for i in range(len(face.points)):
            j = (i + 1) % len(face.points)
            if abs(dg[i]) < TOL_PLAN and abs(dg[j]) < TOL_PLAN:
                ligne = LineString(rg.vers_2d(face.points[[i, j]])).intersection(pg.buffer(0.5))
                if ligne.length > 1.0:
                    for morceau in _lignes(ligne):
                        pts3 = rg.vers_3d(np.asarray(morceau.coords))
                        zones.append(_tampon(LineString(rep.vers_2d(pts3)), ms))
    if face.id == ctx.fixation:
        x0 = float(face.points[0, 0])
        for y, z in ctx.trous_vis:
            zones.append(_tampon(Point(rep.vers_2d([x0, y, z])), MARGE_VIS))
    if face.id == ctx.appui:
        c = np.array(ctx.point_charge, float)
        c = c - rep.distance_plan(c) * rep.n
        zones.append(_tampon(Point(rep.vers_2d(c)), RAYON_APPUI_PROTEGE))
    return unary_union(zones) if zones else Polygon()


def construire(face, specs: list[dict], ctx: Contexte) -> ResultatEvidements:
    """Découpes d'une face. `specs` : évidements déjà évalués (nombres), avec leur « type »."""
    res = ResultatEvidements()
    rep = Repere.de(face.points, face.normale())
    p2 = rep.vers_2d(face.points)
    poly = Polygon(p2)
    if not poly.is_valid or poly.area < 1.0:
        res.erreurs.append(f"{face.id} : contour non simple, impossible d'y découper un évidement")
        return res
    prot = zones_protegees(face, ctx)
    t = face.t
    lig, rmin = ligament_min(t), rayon_min(t)
    candidats: list[tuple[str, Polygon, float]] = []  # trous libres (cercle, polygone) à contrôler
    morceaux: list[tuple[Polygon, float]] = []
    for k, s in enumerate(specs, start=1):
        nom = f"évidement {k} de {face.id} ({s['type']})"
        if s["type"] == "decalage":
            m = s["marge"]
            if m <= 1e-9:
                continue
            r = s.get("rayon") or rayon_defaut(t)
            if m < lig - 1e-6:
                res.erreurs.append(f"{nom} : marge {m:.1f} mm < bande minimale {lig:.1f} mm (max(2 t, 10 mm))")
                continue
            if r < rmin - 1e-6:
                res.erreurs.append(f"{nom} : rayon {r:.1f} mm < rayon minimal {rmin:.1f} mm (max(t, 5 mm))")
                continue
            coupe = [prot]
            w = s.get("largeur_nervure") or lig
            for n_k, nerv in enumerate(s.get("nervures") or [], start=1):
                try:
                    a, b = (_point_bord(p2, int(i), float(u)) for i, u in nerv)
                except (TypeError, ValueError, IndexError):
                    res.erreurs.append(f"{nom} : nervure {n_k} mal décrite (attendu [[arête, position], [arête, position]])")
                    continue
                if w < lig - 1e-6:
                    res.erreurs.append(f"{nom} : nervure de {w:.1f} mm < largeur minimale {lig:.1f} mm")
                    continue
                coupe.append(LineString([a, b]).buffer(w / 2, cap_style="flat"))
            poche = poly.buffer(-m, join_style="mitre", mitre_limit=20.0).difference(unary_union(coupe))
            poche = _relier_ilots(poche, poly, lig)
            poche = _ouverture(poche, r)
            for p in _polygones(poche):
                if p.interiors:
                    res.notes.append(f"{nom} : une partie entoure une zone protégée, elle est laissée pleine")
                elif p.area < AIRE_MIN:
                    res.notes.append(f"{nom} : morceau de {p.area:.0f} mm² ignoré (< {AIRE_MIN:.0f} mm²)")
                else:
                    morceaux.append((p, r))
            if not any(True for _ in _polygones(poche)):
                res.notes.append(f"{nom} : vide (la marge et les zones protégées ne laissent rien à découper)")
        elif s["type"] == "cercle":
            d = s["diametre"]
            if d <= 1e-9:
                continue
            c = np.asarray(s["centre"], float)
            if abs(rep.distance_plan(c)) > max(t, 1.0):
                res.erreurs.append(f"{nom} : le centre est à {abs(rep.distance_plan(c)):.1f} mm du plan de la face")
                continue
            if d / 2 < rmin - 1e-6:
                res.erreurs.append(f"{nom} : diamètre {d:.1f} mm < {2 * rmin:.1f} mm (2 x rayon minimal)")
                continue
            candidats.append((nom, Point(rep.vers_2d(c)).buffer(d / 2, quad_segs=SEGMENTS_QUART), d / 2))
        elif s["type"] == "polygone":
            pts = np.asarray(s["points"], float)
            if len(pts) < 3 or (np.abs(rep.distance_plan(pts)) > max(t, 1.0)).any():
                res.erreurs.append(f"{nom} : il faut au moins 3 points, dans le plan de la face")
                continue
            r = s.get("rayon") or rayon_defaut(t)
            if r < rmin - 1e-6:
                res.erreurs.append(f"{nom} : rayon {r:.1f} mm < rayon minimal {rmin:.1f} mm")
                continue
            p = Polygon(rep.vers_2d(pts))
            if not p.is_valid:
                res.erreurs.append(f"{nom} : contour qui se recoupe")
                continue
            p = _ouverture(p, r)
            if p.is_empty or p.area < AIRE_MIN:
                res.erreurs.append(f"{nom} : trop petit ou trop étroit pour un rayon de {r:.0f} mm")
                continue
            candidats.append((nom, _plus_grand(p), r))
        else:
            res.erreurs.append(f"{nom} : type inconnu (decalage, cercle, polygone)")
    permis = poly.buffer(-lig)
    for nom, h, r in candidats:
        if h.intersects(prot) and h.intersection(prot).area > 1e-3:
            res.erreurs.append(f"{nom} : empiète sur une zone protégée (pli, soudure, vis ou appui)")
        elif not h.difference(permis).area <= 1e-3:
            res.erreurs.append(f"{nom} : à moins de {lig:.0f} mm du bord de la face (bande minimale)")
        else:
            morceaux.append((h, r))
    for i, (a, _) in enumerate(morceaux):
        for b, _ in morceaux[i + 1:]:
            if a.distance(b) < lig - 1e-6:
                res.erreurs.append(f"{face.id} : deux évidements à {a.distance(b):.1f} mm l'un de l'autre "
                                   f"(bande minimale {lig:.0f} mm)")
    if res.erreurs:
        return res
    for p, r in morceaux:
        anneau = np.asarray(p.exterior.coords)[:-1]
        res.trous.append(rep.vers_3d(anneau))
        res.trous_2d.append(p)
        res.rayons.append(r)
    return res


def triangles_avec_trous(points: np.ndarray, normale: np.ndarray, trous: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Triangulation d'une face percée (pour l'affichage 3D) : sommets (k, 3) et triangles (m, 3)."""
    rep = Repere.de(points, normale)
    poly = Polygon(rep.vers_2d(points), [rep.vers_2d(t) for t in trous])
    tri = constrained_delaunay_triangles(poly)
    sommets: list[tuple[float, float]] = []
    index: dict[tuple[float, float], int] = {}
    faces = []
    for tr in tri.geoms:
        ids = []
        for xy in list(tr.exterior.coords)[:3]:
            cle = (round(xy[0], 6), round(xy[1], 6))
            if cle not in index:
                index[cle] = len(sommets)
                sommets.append(cle)
            ids.append(index[cle])
        faces.append(ids)
    return rep.vers_3d(np.array(sommets)), np.array(faces, int)


# ---------- outils ----------
def _tampon(g: BaseGeometry, d: float) -> BaseGeometry:
    """Zone à distance d d'une géométrie. Les arcs sont des polygones circonscrits (rayon d / cos(π / 4n)),
    pour que la distance protégée soit tenue partout et pas seulement aux sommets."""
    return g.buffer(d / math.cos(math.pi / (4 * SEGMENTS_QUART)), quad_segs=SEGMENTS_QUART)


def _ouverture(g: BaseGeometry, r: float) -> BaseGeometry:
    """Ouverture morphologique : arrondit au rayon r les coins saillants de la découpe et retire les parties
    plus étroites que 2 r."""
    if g.is_empty:
        return g
    return g.buffer(-r, join_style="round", quad_segs=SEGMENTS_QUART).buffer(
        r, join_style="round", quad_segs=SEGMENTS_QUART)


def _relier_ilots(poche: BaseGeometry, face: Polygon, lig: float) -> BaseGeometry:
    """Une zone protégée entourée par la poche flotterait : on la relie au bord le plus proche par un bras."""
    for _ in range(6):
        bras = []
        for p in _polygones(poche):
            for anneau in p.interiors:
                ilot = Polygon(anneau)
                _, b = nearest_points(ilot, face.exterior)
                c = ilot.representative_point()
                d = np.array(b.coords[0]) - np.array(c.coords[0])
                if np.linalg.norm(d) < 1e-9:
                    continue
                bras.append(LineString([c, b]).buffer(lig, cap_style="flat"))
        if not bras:
            return poche
        poche = poche.difference(unary_union(bras))
    return poche


def _polygones(g: BaseGeometry):
    if g.is_empty:
        return
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, MultiPolygon):
        yield from g.geoms
    elif hasattr(g, "geoms"):
        for h in g.geoms:
            yield from _polygones(h)


def _plus_grand(g: BaseGeometry) -> Polygon:
    return max(_polygones(g), key=lambda p: p.area)


def _lignes(g: BaseGeometry):
    if g.is_empty:
        return
    if isinstance(g, LineString):
        yield g
    elif hasattr(g, "geoms"):
        for h in g.geoms:
            yield from _lignes(h)


def _point_bord(p2: np.ndarray, i: int, u: float) -> np.ndarray:
    if not 0 <= i < len(p2) or not -1e-9 <= u <= 1 + 1e-9:
        raise ValueError("arête ou position hors limites")
    a, b = p2[i], p2[(i + 1) % len(p2)]
    return a + u * (b - a)


def _segments_communs(f1, f2, tol: float = 1e-6) -> list[np.ndarray]:
    """Segments communs (longueur > 1 mm) entre les arêtes de deux faces : lignes de pli d'une même tôle."""
    segs = []
    p1, p2 = f1.points, f2.points
    for i in range(len(p1)):
        a, b = p1[i], p1[(i + 1) % len(p1)]
        u = b - a
        L = np.linalg.norm(u)
        if L < 1e-9:
            continue
        u = u / L
        for j in range(len(p2)):
            c, d = p2[j], p2[(j + 1) % len(p2)]
            if np.linalg.norm(np.cross(c - a, u)) > tol or np.linalg.norm(np.cross(d - a, u)) > tol:
                continue
            lo = max(0.0, min(np.dot(c - a, u), np.dot(d - a, u)))
            hi = min(L, max(np.dot(c - a, u), np.dot(d - a, u)))
            if hi - lo > 1.0:
                segs.append(np.array([a + lo * u, a + hi * u]))
    return segs


def chaines_contour(anneau: np.ndarray, angle_vif: float = 20.0) -> list[tuple[str, np.ndarray]]:
    """Découpe un contour fermé (k, 3) en morceaux pour la CAO et le maillage : ("ligne", 2 points) pour les
    segments droits, ("spline", points) pour les arcs. Coupure aux angles vifs et aux longs segments droits,
    pour qu'une spline ne passe jamais au large d'un coin ou d'un bord droit."""
    p = np.asarray(anneau, float)
    p = p[np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1) > 1e-6]  # points doublés retirés
    n = len(p)
    seg = np.roll(p, -1, axis=0) - p  # seg[i] : de p[i] à p[i+1]
    L = np.linalg.norm(seg, axis=1)
    u = seg / L[:, None]
    angle = np.degrees(np.arccos(np.clip(np.sum(np.roll(u, 1, axis=0) * u, axis=1), -1, 1)))  # au sommet i
    long = L > 3.0 * np.median(L)
    coupe = (angle > angle_vif) | long | np.roll(long, 1)
    if not coupe.any():
        return [("spline_fermee", p)]
    debut = int(np.argmax(coupe))
    ordre = [(debut + k) % n for k in range(n + 1)]
    morceaux, courant = [], [ordre[0]]
    for i in ordre[1:]:
        courant.append(i)
        if coupe[i]:
            morceaux.append(courant)
            courant = [i]
    sortie = []
    for m in morceaux:
        pts = p[m]
        if len(m) == 2 or (angle[m[1:-1]] < 0.5).all():
            sortie.append(("ligne", pts[[0, -1]]))
        else:
            sortie.append(("spline", pts))
    return sortie


def rayon_courbure_min(anneau: np.ndarray) -> float:
    """Plus petit rayon de courbure d'un contour discrétisé (cercle circonscrit à trois points voisins)."""
    p = np.asarray(anneau, float)
    a, b, c = np.roll(p, 1, axis=0), p, np.roll(p, -1, axis=0)
    ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
    aire2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ab * bc * ca / (2 * aire2)
    r = r[np.isfinite(r) & (aire2 > 1e-9)]
    return float(r.min()) if len(r) else math.inf
