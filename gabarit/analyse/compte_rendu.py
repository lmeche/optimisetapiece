"""Compte rendu chiffré d'un concept : où passe l'effort, quelle matière travaille, pourquoi ce kg/kN.

Décomposition exacte de l'objectif (identité, pas une approximation) :

    kg/kN = 1000 . rho / Re . L_eq / eta

- L_eq, longueur équivalente du chemin d'effort (mm) = somme sur les éléments de V . sigma_moy / F : ce qu'il
  faut de « contrainte x volume » pour porter l'effort F. Pour un tirant de longueur L, L_eq = L ; pour une
  barre de treillis, c'est la somme des |N| L / F (quantité de Michell). Elle ne dépend que du chemin d'effort :
  court et direct (traction, compression) = petit ; détours et flexion = grand.
- eta, taux d'utilisation de la matière = masse x sigma_moy (moyenne pondérée par la masse) / (M . sigma_utile) :
  1 si toute la matière travaillait à la contrainte du point critique ; une tôle en flexion pure plafonne à 0,5
  (le plan moyen ne travaille pas), une zone peu chargée le fait baisser.
Pour comparer deux concepts : le rapport des kg/kN est le produit du rapport des L_eq et de l'inverse du
rapport des eta, et chaque L_eq se répartit sur les faces.

sigma_moy : Von Mises moyen dans le volume de l'élément ; sigma_utile : contrainte maximale hors singularités
(bord des vis, sous l'appui, extrémités de jonction), celle qui fixe l'effort tenu (voir gabarit/evaluation.py).
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from gabarit.analyse.champ import ChampElements, champ_elements
from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts import evidements as ev
from gabarit.evaluation import evaluer, singulier

SEUIL_PEU_UTILE = 0.2  # une peau sous 20 % de sigma_utile : matière qui ne sert presque pas
FLAMBEMENT_MIN = 3.0
FT_RD_M10_88 = 0.9 * 800 * 58.0 / 1.25  # N, résistance de calcul d'une vis M10 8.8 en traction
COLONNES_CARTE = 28
NIVEAUX = "0123456789"  # dizaine de % de sigma_utile ; « ! » au-delà de 100 %
LEGENDE_CARTES = ("Cartes d'utilisation : chiffre = dizaine de % de la contrainte utile en peau (0 : moins de 10 %, "
                  "9 : 90 % et plus, ! : au-delà), ~ = zone exclue du critère (autour des vis, sous l'appui, au bout "
                  "des jonctions), espace = pas de tôle, X = point critique.")


@dataclass
class BilanFace:
    id: str
    t: float
    aire_mm2: float
    masse_kg: float
    part_masse: float
    part_energie: float
    l_eq_mm: float  # contribution de la face à L_eq
    eta: float  # taux d'utilisation de la matière de la face
    part_flexion: float  # part de l'énergie de la face stockée en flexion
    sigma_peau_max: float  # MPa, hors singularités
    part_peu_utile: float  # part de la surface dont les peaux restent sous SEUIL_PEU_UTILE x sigma_utile
    carte: list[str] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)  # légende des axes de la carte : horizontal, vertical


@dataclass
class CompteRendu:
    concept: str
    valeurs: dict
    taille: float
    valide: bool
    erreurs: list[str] = field(default_factory=list)
    mesures: dict = field(default_factory=dict)
    marges: dict = field(default_factory=dict)  # rapport à la limite (> 1 : respectée)
    limite: str = ""
    l_eq_mm: float = math.nan
    eta: float = math.nan
    part_flexion: float = math.nan
    point_critique: dict = field(default_factory=dict)
    faces: list[BilanFace] = field(default_factory=list)
    flambement: dict = field(default_factory=dict)
    pistes: list[str] = field(default_factory=list)
    controle: dict = field(default_factory=dict)
    dossier: str = ""

    def en_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def depuis_dict(cls, d: dict) -> CompteRendu:
        d = dict(d)
        d["faces"] = [BilanFace(**f) for f in d.get("faces", [])]
        return cls(**d)

    def face(self, ident: str) -> BilanFace:
        return next(f for f in self.faces if f.id == ident)

    # ---------- textes ----------
    def resume(self) -> str:
        """Une ligne : le concept, son kg/kN et sa décomposition."""
        if not self.valide:
            return f"{self.concept} : invalide ({'; '.join(self.erreurs[:2])})"
        m = self.mesures
        return (f"{self.concept} : {m['kg_par_kN']:.3f} kg/kN (masse {m['masse_kg']:.2f} kg, effort tenu "
                f"{m['effort_tenu_N'] / 1000:.1f} kN, flambement {_f(m['flambement'], 2)}), L_eq {self.l_eq_mm:.0f} mm, "
                f"eta {self.eta:.3f}, limite : {self.limite}")

    def texte(self, cartes: int = 99, faces_max: int = 12) -> str:
        """Compte rendu compact, pour une IA. cartes : nombre de cartes d'utilisation (faces les plus chargées)."""
        if not self.valide:
            return f"## Compte rendu : {self.concept}\nInvalide : " + "; ".join(self.erreurs)
        m, g = self.mesures, self.marges
        lignes = [
            f"## Compte rendu : {self.concept}",
            f"Calcul coques S8R, maille {self.taille:g} mm, {m['effort_N']:.0f} N sur l'appui.",
            (f"Masse {m['masse_kg']:.3f} kg ; effort tenu {m['effort_tenu_N'] / 1000:.2f} kN (sigma utile "
             f"{m['sigma_utile']:.0f} MPa) ; {m['kg_par_kN']:.4f} kg/kN ; flèche {m['fleche']:.2f} mm ; flambement "
             f"{_f(m['flambement'], 2)} ; traction max par vis {m['traction_vis_N'] / 1000:.1f} kN."),
            (f"Marges (> 1 = tenu) : plastification x{g['plastification']:.2f}, flambement x{_f(g['flambement'], 2)} "
             f"(mini {FLAMBEMENT_MIN:g}), vis x{g['vis']:.2f}. Ce qui limite : {self.limite}."),
            (f"Décomposition : kg/kN = 1000 rho/Re x L_eq / eta = {1000 * m['rho'] / m['Re']:.3e} x "
             f"{self.l_eq_mm:.0f} / {self.eta:.3f} (L_eq : longueur équivalente du chemin d'effort, somme V.sigma/F ; "
             f"eta : taux d'utilisation de la matière ; énergie stockée en flexion : {100 * self.part_flexion:.0f} %)."),
            f"Point critique : {self.point_critique.get('texte', '?')}",
        ]
        if self.flambement:
            fl = self.flambement
            lignes.append(f"Mode de flambement retenu (facteur {fl['facteur']:.2f}) : faces qui bougent "
                          + ", ".join(f"{f} {100 * p:.0f} %" for f, p in fl["faces"][:5]) + ".")
        lignes.append("\nFaces (part de masse, part d'énergie, L_eq, eta, part d'énergie en flexion, peau max en % de "
                      f"sigma utile, part de surface sous {100 * SEUIL_PEU_UTILE:.0f} %) :")
        for f in sorted(self.faces, key=lambda f: -f.l_eq_mm)[:faces_max]:
            lignes.append(
                f"- {f.id} (t {f.t:g}) : masse {100 * f.part_masse:.0f} %, énergie {100 * f.part_energie:.0f} %, "
                f"L_eq {f.l_eq_mm:.0f} mm, eta {f.eta:.2f}, flexion {100 * f.part_flexion:.0f} %, peau max "
                f"{100 * f.sigma_peau_max / m['sigma_utile']:.0f} %, peu utile {100 * f.part_peu_utile:.0f} %")
        if self.pistes:
            lignes.append("\nPistes (règles automatiques, à vérifier par le calcul) :")
            lignes += [f"- {p}" for p in self.pistes]
        a_cartographier = sorted(self.faces, key=lambda f: -f.part_energie)[:cartes]
        if a_cartographier:
            lignes.append("\n" + LEGENDE_CARTES)
            for f in a_cartographier:
                lignes.append(f"{f.id} ({f.axes[0]} ; {f.axes[1]}) :")
                lignes += [f"  {ligne}" for ligne in f.carte]
        return "\n".join(lignes)

    def markdown(self) -> str:
        if not self.valide:
            return f"# Compte rendu : {self.concept}\n\nConcept invalide :\n\n" + "\n".join(f"- {e}" for e in self.erreurs)
        m, g = self.mesures, self.marges
        md = [f"# Compte rendu : {self.concept}", "",
              f"Calcul coques S8R, maille {self.taille:g} mm, {m['effort_N']:.0f} N sur l'appui.", "",
              "| Mesure | Valeur |", "| --- | --- |",
              f"| Masse | {m['masse_kg']:.3f} kg |",
              f"| Effort tenu (première plastification) | {m['effort_tenu_N'] / 1000:.2f} kN |",
              f"| Objectif | **{m['kg_par_kN']:.4f} kg/kN** |",
              f"| Contrainte utile à {m['effort_N']:.0f} N | {m['sigma_utile']:.0f} MPa |",
              f"| Flèche sous l'appui | {m['fleche']:.2f} mm |",
              f"| Facteur de flambement | {_f(m['flambement'], 2)} |",
              f"| Traction max par vis | {m['traction_vis_N'] / 1000:.1f} kN |", "",
              (f"Marges (> 1 = tenu) : plastification x{g['plastification']:.2f}, flambement "
               f"x{_f(g['flambement'], 2)}, vis x{g['vis']:.2f}. **Ce qui limite : {self.limite}.**"), "",
              "## Pourquoi ce kg/kN", "",
              (f"kg/kN = 1000 rho / Re x L_eq / eta = {1000 * m['rho'] / m['Re']:.3e} x {self.l_eq_mm:.0f} mm / "
               f"{self.eta:.3f}, avec L_eq la longueur équivalente du chemin d'effort (somme des V.sigma / F) et eta "
               f"le taux d'utilisation de la matière. {100 * self.part_flexion:.0f} % de l'énergie de déformation est "
               "stockée en flexion."), "",
              f"Point critique : {self.point_critique.get('texte', '?')}", ""]
        if self.flambement:
            fl = self.flambement
            md += [f"Mode de flambement retenu (facteur {fl['facteur']:.2f}) : "
                   + ", ".join(f"{f} {100 * p:.0f} %" for f, p in fl["faces"][:6]) + ".", ""]
        md += ["## Faces", "",
               "| Face | t | Masse | Énergie | L_eq | eta | Flexion | Peau max | Peu utile |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for f in sorted(self.faces, key=lambda f: -f.l_eq_mm):
            md.append(f"| {f.id} | {f.t:g} | {f.masse_kg:.3f} kg ({100 * f.part_masse:.0f} %) | "
                      f"{100 * f.part_energie:.0f} % | {f.l_eq_mm:.0f} mm | {f.eta:.2f} | {100 * f.part_flexion:.0f} % | "
                      f"{f.sigma_peau_max:.0f} MPa | {100 * f.part_peu_utile:.0f} % |")
        if self.pistes:
            md += ["", "## Pistes (règles automatiques)", ""] + [f"- {p}" for p in self.pistes]
        md += ["", "## Cartes d'utilisation", "", LEGENDE_CARTES, ""]
        for f in sorted(self.faces, key=lambda f: -f.part_energie):
            md += [f"**{f.id}** ({f.axes[0]} ; {f.axes[1]})", "", "```"] + f.carte + ["```", ""]
        c = self.controle
        md += ["## Contrôles", "",
               (f"Énergie de déformation des éléments {c['energie_Nmm']:.1f} N.mm pour un travail de l'effort de "
                f"{c['travail_Nmm']:.1f} N.mm (écart {c['ecart_energie_pct']:+.3f} %). Masse maillée "
                f"{c['masse_maillee_kg']:.3f} kg pour {m['masse_kg']:.3f} kg. Identité 1000 rho/Re x L_eq/eta = "
                f"{c['identite_kg_par_kN']:.4f} kg/kN. Résultante des efforts et des réactions "
                f"{c.get('equilibre', math.nan):.1e} de l'effort"
                + (f" (solution obtenue au calcul n° {c['essais_calcul']}, les précédents étant faux)"
                   if c.get("essais_calcul", 1) > 1 else "") + ".")]
        return "\n".join(md)


def analyser(concept, cdc: CahierDesCharges, taille: float = 6.0, flambement: bool = True,
             dossier: Path | None = None) -> CompteRendu:
    """Calcule le concept (statique avec contact, flambement) et en tire le compte rendu."""
    nom = concept.definition.get("nom", "concept") if hasattr(concept, "definition") else type(concept).__name__
    dossier = Path(dossier or tempfile.mkdtemp(prefix="gabarit_cr_"))
    garde: dict = {}
    r = evaluer(concept, cdc, taille=taille, dossier=dossier, flambement=flambement, garde=garde)
    cr = CompteRendu(nom, dict(concept.parametres()), taille, r.valide, list(r.erreurs), dossier=str(dossier))
    if not r.valide:
        return cr
    geo = concept.geometrie()
    faces = {f.id: f for f in geo.faces}
    modele, info, res = garde["modele"], garde["info"], garde["statique"]
    champ = champ_elements(modele, garde["faces_elements"], res, {i: f.t for i, f in faces.items()})
    F = cdc.chargements[0].force
    rho = cdc.materiau.rho * 1e-9  # kg/mm³
    Re = cdc.materiau.Re
    sigma_ref = r.sigma_max_utile

    cr.mesures = {"masse_kg": r.masse_kg, "effort_tenu_N": r.effort_tenu, "kg_par_kN": r.masse_par_kN,
                  "sigma_utile": sigma_ref, "fleche": r.fleche, "flambement": r.facteur_flambement,
                  "traction_vis_N": r.traction_vis_max, "effort_N": F, "rho": rho, "Re": Re}
    cr.marges = {"plastification": r.effort_tenu / F,
                 "flambement": r.facteur_flambement / FLAMBEMENT_MIN if math.isfinite(r.facteur_flambement) else math.nan,
                 "vis": FT_RD_M10_88 / max(r.traction_vis_max, 1e-9)}
    cr.limite = _limite(cr.marges, flambement)

    vs = champ.volume * champ.vm_moyen  # N.mm / mm... : V.sigma
    cr.l_eq_mm = float(vs.sum() / F)
    cr.eta = float(rho * vs.sum() / (r.masse_kg * sigma_ref))
    cr.part_flexion = float((champ.energie * champ.part_flexion).sum() / champ.energie.sum())
    singulier = _elements_singuliers(champ, info)

    p_crit = np.asarray(res.coords[garde["n_pic"]], float)
    face_crit = _face_critique(champ, p_crit, taille)
    cr.point_critique = _point_critique(p_crit, sigma_ref, face_crit, faces, geo, info)

    for fid, f in faces.items():
        w = champ.poids_face(fid)
        if not w.any():
            continue
        m_f = rho * float((w * champ.volume).sum())
        dans = w > 0
        ok = dans & ~singulier
        aire_f = float((w * champ.aire).sum())
        b = BilanFace(
            id=fid, t=f.t, aire_mm2=round(f.aire_nette(), 1), masse_kg=m_f,
            part_masse=m_f / (rho * float(champ.volume.sum())),
            part_energie=float((w * champ.energie).sum() / champ.energie.sum()),
            l_eq_mm=float((w * vs).sum() / F),
            eta=float(rho * (w * vs).sum() / (m_f * sigma_ref)) if m_f > 0 else math.nan,
            part_flexion=float((w * champ.energie * champ.part_flexion).sum() / max((w * champ.energie).sum(), 1e-30)),
            sigma_peau_max=float(champ.vm_peau[ok].max()) if ok.any() else float(champ.vm_peau[dans].max()),
            part_peu_utile=float((w * champ.aire * (champ.vm_peau < SEUIL_PEU_UTILE * sigma_ref)).sum() / aire_f),
        )
        b.carte, b.axes = _carte(f, champ, dans, sigma_ref, p_crit if face_crit == fid else None, singulier)
        cr.faces.append(b)

    if flambement and garde.get("flambement") is not None and math.isfinite(r.facteur_flambement):
        cr.flambement = _mode(garde, modele, champ, r.facteur_flambement)
    travail = 0.5 * sum(float(np.dot(modele.forces[n], res.u_noeuds[n])) for n in modele.forces)
    cr.controle = {"energie_Nmm": float(champ.energie.sum()), "travail_Nmm": travail,
                   "ecart_energie_pct": 100 * (float(champ.energie.sum()) / travail - 1) if travail else math.nan,
                   "masse_maillee_kg": rho * float(champ.volume.sum()),
                   "identite_kg_par_kN": 1000 * rho * cr.l_eq_mm / (cr.eta * Re),
                   "equilibre": (res.bilan or {}).get("equilibre", math.nan), "essais_calcul": res.essais}
    cr.pistes = pistes(cr, geo)
    return cr


# ---------- détails ----------
def _limite(marges: dict, flambement: bool) -> str:
    noms = {"plastification": "la plastification (contrainte au point critique)",
            "flambement": "le flambement", "vis": "la traction dans les vis"}
    ok = {k: v for k, v in marges.items() if math.isfinite(v) and (flambement or k != "flambement")}
    k = min(ok, key=ok.get)
    if ok[k] < 1:
        return f"{noms[k]}, NON TENU (marge x{ok[k]:.2f})"
    return noms[k] + ("" if flambement else " (flambement non calculé)")


def _elements_singuliers(champ: ChampElements, info: dict) -> np.ndarray:
    """Éléments des zones exclues de sigma_utile : anneaux des vis, disque sous l'appui, extrémités de jonction
    (mêmes règles que gabarit/evaluation.py, appliquées au centre des éléments)."""
    return np.array([singulier(p, info) for p in champ.centre], bool)


def _face_critique(champ: ChampElements, p: np.ndarray, taille: float) -> str:
    """Face de l'élément le plus sollicité autour du point critique (à une jonction, celle qui travaille)."""
    d = np.linalg.norm(champ.centre - p, axis=1)
    proches = np.flatnonzero(d <= max(1.5 * taille, d.min() + 1e-9))
    e = proches[np.argmax(champ.vm_peau[proches])]
    return max(champ.faces[e], key=lambda f: champ.poids_face(f)[e]) if champ.faces[e] else "?"


def _point_critique(p: np.ndarray, sigma: float, face_id: str, faces: dict, geo, info: dict) -> dict:
    f = faces.get(face_id)
    texte = f"{sigma:.0f} MPa en (x {p[0]:.0f}, y {p[1]:.0f}, z {p[2]:.0f})"
    if f is None:
        return {"face": face_id, "xyz": p.round(1).tolist(), "sigma": sigma, "voisinage": [], "texte": texte}
    vois = voisinage(f, geo.faces, p, info, geo)
    texte = f"{sigma:.0f} MPa sur {face_id} en (x {p[0]:.0f}, y {p[1]:.0f}, z {p[2]:.0f})"
    if vois:
        texte += ", " + ", ".join(v for _, v in vois[:3])
    else:
        texte += ", en pleine tôle (loin des soudures, plis, bords et trous)"
    return {"face": face_id, "xyz": p.round(1).tolist(), "sigma": sigma, "voisinage": [v for _, v in vois],
            "texte": texte}


def voisinage(face, faces: list, p, info: dict, geo=None, rayon: float | None = None) -> list[tuple[float, str]]:
    """Ce qui entoure un point d'une face (soudure, extrémité de soudure, pli, doublage, évidement, bord libre,
    vis, appui), trié par distance. rayon : distance maximale (défaut max(3 t, 15 mm))."""
    rayon = rayon or max(3 * face.t, 15.0)
    rep = ev.Repere.de(face.points, face.normale())
    q = Point(rep.vers_2d(np.asarray(p, float) - rep.distance_plan(p) * rep.n))
    poly = Polygon(rep.vers_2d(face.points))
    sortie: list[tuple[float, str]] = []
    lignes = []
    for nature, g, geo2 in ev.liaisons(face, faces):
        if nature == "doublage":
            d = geo2.exterior.distance(q) if hasattr(geo2, "exterior") else geo2.boundary.distance(q)
            sortie.append((d, f"à {d:.0f} mm du bord du doublage avec {g.id}"))
            continue
        lignes.append(geo2)
        d = geo2.distance(q)
        quoi = "du pli avec" if nature == "pli" else "de la soudure avec"
        texte = f"à {d:.0f} mm {quoi} {g.id}"
        if nature == "soudure":  # une extrémité de soudure à l'intérieur de la face concentre l'effort
            bouts = [Point(c) for c in (geo2.coords[0], geo2.coords[-1]) if poly.exterior.distance(Point(c)) > 1.0]
            if bouts:
                db = min(b.distance(q) for b in bouts)
                if db <= rayon:
                    texte += f" (extrémité de la soudure à {db:.0f} mm)"
        sortie.append((d, texte))
    for k, trou in enumerate(face.trous, start=1):
        d = Polygon(rep.vers_2d(np.asarray(trou))).exterior.distance(q)
        sortie.append((d, f"à {d:.0f} mm du bord de l'évidement {k}"))
    libre = poly.exterior
    if lignes:
        libre = libre.difference(unary_union([ligne.buffer(0.5) for ligne in lignes]))
    if not libre.is_empty:
        d = libre.distance(q)
        sortie.append((d, f"à {d:.0f} mm d'un bord libre"))
    if geo is not None and face.id == geo.fixation:
        x0 = float(face.points[0, 0])
        for y, z in info.get("trous", []):
            d = max(0.0, Point(rep.vers_2d([x0, y, z])).distance(q) - 8.0)
            sortie.append((d, f"à {d:.0f} mm de la zone serrée d'une vis (y {y:.0f}, z {z:.0f})"))
    if geo is not None and face.id == geo.appui:
        c = np.array([info["x_charge"], 0.0, info["z_charge"]])
        d = max(0.0, Point(rep.vers_2d(c - rep.distance_plan(c) * rep.n)).distance(q) - 25.0)
        sortie.append((d, f"à {d:.0f} mm du bord de l'appui"))
    return sorted([s for s in sortie if s[0] <= rayon], key=lambda s: s[0])


def repere_lisible(face) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Repère de carte d'une face aligné sur les axes du socle quand c'est possible : origine, axe horizontal,
    axe vertical de la carte, et légende de ces deux axes."""
    n = face.normale()
    X, Y, Z = np.eye(3)
    if abs(n[2]) > 0.95:  # face horizontale : x vers la droite, y vers le haut de la carte
        e1 = X - np.dot(X, n) * n
    else:  # sinon : le haut de la carte monte dans la face
        e2 = Z - np.dot(Z, n) * n
        e2 /= np.linalg.norm(e2)
        e1 = np.cross(e2, n)
    e1 /= np.linalg.norm(e1)
    if np.dot(e1, X) < -1e-9 or (abs(np.dot(e1, X)) < 1e-9 and np.dot(e1, Y) < 0):
        e1 = -e1
    e2 = np.cross(n, e1)
    if np.dot(e2, Z) < -1e-9 or (abs(np.dot(e2, Z)) < 1e-9 and np.dot(e2, Y) < 0):
        e2 = -e2
    o = face.points.min(axis=0)
    legendes = [_legende(face.points, e1, "→"), _legende(face.points, e2, "↑")]
    return o, e1, e2, legendes


def _legende(points: np.ndarray, a: np.ndarray, fleche: str) -> str:
    s = points @ a
    lo, hi = points[np.argmin(s)], points[np.argmax(s)]
    k = int(np.argmax(np.abs(a)))
    nom = "xyz"[k]
    if abs(a[k]) > 0.99:
        return f"{fleche} {nom} de {lo[k]:.0f} à {hi[k]:.0f}"
    autres = [j for j in range(3) if abs(a[j]) > 0.1]
    debut = ", ".join(f"{'xyz'[j]} {lo[j]:.0f}" for j in autres)
    fin = ", ".join(f"{'xyz'[j]} {hi[j]:.0f}" for j in autres)
    return f"{fleche} en biais, de ({debut}) à ({fin})"


def _carte(face, champ: ChampElements, dans: np.ndarray, sigma_ref: float, p_crit,
           exclus: np.ndarray) -> tuple[list[str], list[str]]:
    """Carte d'utilisation d'une face (peau en dizaines de % de sigma_utile)."""
    idx = np.flatnonzero(dans)
    u = champ.vm_peau[idx] / sigma_ref
    codes = np.where(exclus[idx], -1, np.where(u >= 1.0 + 1e-9, 10, np.minimum(9, (10 * u).astype(int))))
    symboles = {-1: "~", 10: "!", **{k: NIVEAUX[k] for k in range(10)}}
    return carte_face(face, champ.centre[idx], codes, symboles, p_crit)


def carte_face(face, centres: np.ndarray, codes: np.ndarray, symboles: dict[int, str],
               marque=None) -> tuple[list[str], list[str]]:
    """Carte texte d'une face : chaque case montre le plus grand code des éléments dont le centre y tombe (celui
    de l'élément le plus proche si elle n'en contient aucun), un espace hors de la tôle, X au point `marque`.
    Une case est deux fois plus haute que large, comme un caractère. Renvoie les lignes et la légende des axes."""
    o, e1, e2, legendes = repere_lisible(face)
    p2 = np.stack([(face.points - o) @ e1, (face.points - o) @ e2], axis=1)
    matiere = Polygon(p2, [np.stack([(np.asarray(t) - o) @ e1, (np.asarray(t) - o) @ e2], axis=1) for t in face.trous])
    (x0, y0), (x1, y1) = p2.min(axis=0), p2.max(axis=0)
    L1, L2 = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    w = max(L1, L2) / COLONNES_CARTE
    nc, nl = max(1, round(L1 / w)), max(1, round(L2 / (2 * w)))
    wc, hc = L1 / nc, L2 / nl  # pavage exact du rectangle englobant
    c2 = np.stack([(centres - o) @ e1, (centres - o) @ e2], axis=1).reshape(-1, 2)
    codes = np.asarray(codes, int)
    arbre = cKDTree(c2) if len(c2) else None
    vide = -(10**6)
    grille = np.full((nl, nc), vide)
    if len(c2):
        col = np.clip(((c2[:, 0] - x0) / wc).astype(int), 0, nc - 1)
        lig = np.clip(((c2[:, 1] - y0) / hc).astype(int), 0, nl - 1)
        np.maximum.at(grille, (lig, col), codes)
    repere = None
    if marque is not None:
        q = np.asarray(marque) - o
        repere = (min(nl - 1, max(0, int(((q @ e2) - y0) / hc))), min(nc - 1, max(0, int(((q @ e1) - x0) / wc))))
    zone = matiere.buffer(0.25 * min(wc, hc))
    lignes = []
    for i in range(nl - 1, -1, -1):  # le haut de la carte d'abord
        s = []
        for j in range(nc):
            v = grille[i, j]
            if v == vide:
                centre = Point(x0 + (j + 0.5) * wc, y0 + (i + 0.5) * hc)
                if arbre is None or not zone.contains(centre):
                    s.append(" ")
                    continue
                v = codes[arbre.query([centre.x, centre.y])[1]]
            s.append("X" if repere == (i, j) else symboles[int(v)])
        lignes.append("".join(s).rstrip())
    return lignes, legendes


def _mode(garde: dict, modele, champ: ChampElements, facteur: float) -> dict:
    """Faces qui bougent dans le mode de flambement retenu (déplacement maximal de la face / maximal global)."""
    rb = garde["flambement"]
    k = int(np.argmin([abs(f - facteur) if f > 0 else math.inf for f in rb.facteurs_flambement]))
    if k >= len(rb.modes):
        return {}
    u = {n: float(np.linalg.norm(v)) for n, v in rb.modes[k].items() if n not in modele.encastres}
    umax = max(u.values()) or 1.0
    par_face: dict[str, float] = {}
    for i, e in enumerate(champ.ids):
        a = max(u.get(n, 0.0) for n in modele.elements[int(e)])
        for f in champ.faces[i]:
            par_face[f] = max(par_face.get(f, 0.0), a / umax)
    faces = sorted(((f, round(p, 3)) for f, p in par_face.items() if p >= 0.1), key=lambda x: -x[1])
    return {"facteur": facteur, "numero_mode": k + 1, "faces": faces}


def pistes(cr: CompteRendu, geo=None) -> list[str]:
    """Règles simples et explicites, tirées des chiffres du compte rendu (ce sont des pistes, pas des preuves)."""
    out = []
    m, g = cr.mesures, cr.marges
    if math.isfinite(g.get("flambement", math.nan)) and g["flambement"] < 1.15:
        faces = ", ".join(f for f, p in cr.flambement.get("faces", [])[:3] if p >= 0.5) or "?"
        out.append(f"Flambement proche de la limite (facteur {m['flambement']:.2f}) : il bloque toute réduction de masse. "
                   f"Faces qui flambent : {faces}. Un retour (pli) sur leur bord libre, une épaisseur plus forte ou une "
                   "largeur plus faible entre appuis les raidissent.")
    if g.get("vis", 9) < 1.3:
        out.append(f"Vis très chargées ({m['traction_vis_N'] / 1000:.1f} kN) : écarter les vis ou en utiliser plus "
                   "(bras de levier plus grand sur le socle).")
    pc = cr.point_critique
    if any("extrémité de la soudure" in v for v in pc.get("voisinage", [])):
        out.append(f"Point critique à l'extrémité d'une ligne de soudure ({pc['face']}) : prolonger la soudure jusqu'à un "
                   "bord, la fermer par un retour, ou élargir la tôle soudée pour étaler l'effort.")
    elif any("du pli" in v for v in pc.get("voisinage", [])[:1]):
        out.append(f"Point critique au pli ({pc['face']}) : la tôle plie le long de ce pli ; un appui (diagonale, "
                   "gousset) ou un retour qui ferme l'angle soulage le pli.")
    for f in sorted(cr.faces, key=lambda f: -f.masse_kg):
        if f.part_masse >= 0.08 and f.part_peu_utile >= 0.5:
            out.append(f"{f.id} : {100 * f.part_peu_utile:.0f} % de sa surface travaille à moins de "
                       f"{100 * SEUIL_PEU_UTILE:.0f} % ({100 * f.part_masse:.0f} % de la masse) : candidate pour un "
                       "évidement (variable « evidement ») ou une tôle plus mince.")
        if f.part_energie >= 0.15 and f.part_flexion >= 0.7:
            out.append(f"{f.id} : {100 * f.part_flexion:.0f} % de son énergie est en flexion pour {100 * f.part_energie:.0f} % "
                       f"de l'énergie totale (L_eq {f.l_eq_mm:.0f} mm) : un appui intermédiaire, une nervure, un retour "
                       "dans le sens de la flexion, ou une forme qui la fait travailler en traction ou compression "
                       "raccourcit le chemin d'effort.")
    return out


def comparer(a: CompteRendu, b: CompteRendu) -> str:
    """Pourquoi a et b diffèrent en kg/kN : rapport des L_eq, rapport des eta, faces qui font la différence."""
    if not (a.valide and b.valide):
        return "comparaison impossible : un des concepts est invalide"
    r = a.mesures["kg_par_kN"] / b.mesures["kg_par_kN"]
    rl, re = a.l_eq_mm / b.l_eq_mm, a.eta / b.eta
    lignes = [f"{a.concept} / {b.concept} : kg/kN x{r:.3f} = L_eq x{rl:.3f} / eta x{re:.3f}."]
    if abs(math.log(rl)) >= abs(math.log(1 / re)):
        lignes.append("L'écart vient surtout du chemin d'effort (L_eq)." if abs(rl - 1) > 0.02 else
                      "Chemins d'effort et utilisation de la matière très proches.")
    else:
        lignes.append("L'écart vient surtout de l'utilisation de la matière (eta).")
    fa = {f.id: f for f in a.faces}
    fb = {f.id: f for f in b.faces}
    ecarts = []
    for fid in set(fa) | set(fb):
        la = fa[fid].l_eq_mm if fid in fa else 0.0
        lb = fb[fid].l_eq_mm if fid in fb else 0.0
        ecarts.append((la - lb, fid, la, lb))
    ecarts.sort(key=lambda x: -abs(x[0]))
    lignes.append("Faces qui pèsent le plus dans l'écart de L_eq : " + "; ".join(
        f"{fid} {la:.0f} contre {lb:.0f} mm" for _, fid, la, lb in ecarts[:4]) + ".")
    return " ".join(lignes)


def _f(x: float, n: int) -> str:
    return f"{x:.{n}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "non calculé"
