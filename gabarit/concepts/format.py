"""Format déclaratif des concepts (phase 4) : ce que l'IA Concepteur écrit, et ce que le calcul consomme.

Un concept est un fichier JSON :

    {
      "nom": "...", "description": "...",
      "variables": {                       # ce que l'optimiseur fait varier
        "t_L": {"type": "epaisseur"},                         # épaisseur de l'atelier
        "z_dessus": {"type": "reel", "min": 100, "max": 200}, # réel borné (mm)
        "u_b": {"type": "fraction"},                          # réel dans [0, 1], à mettre à l'échelle dans "derivees"
        "ecart_vis": {"type": "choix", "valeurs": [0, 20, 40]},
        "h_retour": {"type": "retour", "tole": "L", "max": 60}, # 0 (pas de pli) ou [aile mini de la tôle, max]
        "m_dos": {"type": "evidement", "tole": "L", "max": 50}  # 0 (pas d'évidement) ou [bande mini, max]
      },
      "derivees": {"b": "44 + u_b * 100", ...},               # expressions évaluées dans l'ordre
      "toles": [
        {"nom": "L", "epaisseur": "t_L",
         "faces": {"dos": [[x, y, z], ...], "dessus": [...],  # polygones plans, surfaces moyennes
                   "option": {"points": [...], "si": "h_opt"}},  # face présente seulement si h_opt > 0
         "retours": [{"face": "dessus", "arete": [1, 2], "hauteur": "h_retour", "direction": [0, 0, -1]}],
         "evidements": [{"face": "dos", "type": "decalage", "marge": "m_dos"}]}   # découpes d'allègement
      ],
      "soudures": [["V.diag", "L.dessus"]],                   # liaisons entre tôles, vérifiées (arête commune ou doublage)
      "fixation": {"face": "L.dos", "colonnes_vis": "ecart_vis"},
      "appui": {"face": "L.dessus"},
      "justifications": {"dos": "pourquoi ce choix", ...}
    }

Règles de modélisation :
- deux faces d'une même tôle qui partagent une arête forment un pli (angle et aile minimale vérifiés) ;
- des tôles qui se touchent sont soudées (le maillage les relie) ;
- deux faces coplanaires superposées forment un doublage : leurs épaisseurs s'additionnent.
- un évidement (« decalage », « cercle » ou « polygone ») perce une face : voir le détail plus bas.
Fonctions utilisables dans les expressions : min, max, abs, sqrt, sin, cos, tan, atan2, radians, degrees,
aile_min(t) ; constantes : X_FACE (20), X_CHARGE (320), R_APPUI (25), Y_MAX (80), X_MAX (350).
"""
from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts import evidements as ev
from gabarit.fem.calculix import Materiau, ModeleCoque

X_FACE = 20.0
RAYON_APPUI_VIS = 8.0
MARGE_TROU = 12.0
ANGLE_PLI_MIN = 30.0  # degrés, angle intérieur minimal (hypothèse : outillage pour plis aigus disponible)
DEGAGEMENT_SEMELLE = 10.0

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def vé_et_aile(t: float, ves: list[float]) -> tuple[float, float]:
    ves = ves or [6, 8, 10, 12, 16, 20, 25, 32, 40]
    v = min(ves, key=lambda v: abs(v - 8 * t))
    return v, 0.7 * v


def evaluer_expr(expr, env: dict) -> float:
    """Évalue une expression arithmétique sûre (pas d'appel arbitraire)."""
    if isinstance(expr, (int, float)):
        return float(expr)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise ValueError(f"nom inconnu dans l'expression : {n.id}")
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and callable(env.get(n.func.id)):
            return env[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"expression non autorisée : {ast.dump(n)[:80]}")

    return float(ev(ast.parse(str(expr), mode="eval")))


@dataclass
class Face:
    tole: str
    nom: str
    t: float
    points: np.ndarray  # (n, 3)
    trous: list = field(default_factory=list)  # contours 3D des évidements, (k, 3) chacun
    aire_trous: float = 0.0  # mm²
    rayons_trous: list = field(default_factory=list)  # rayon de courbure minimal de chaque évidement (maillage)

    @property
    def id(self) -> str:
        return f"{self.tole}.{self.nom}"

    def normale(self) -> np.ndarray:
        p = self.points
        n = np.zeros(3)
        for i in range(len(p)):  # méthode de Newell
            a, b = p[i], p[(i + 1) % len(p)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        return n / (np.linalg.norm(n) or 1)

    def aire(self) -> float:
        p = self.points
        s = np.zeros(3)
        for i in range(len(p)):
            s += np.cross(p[i], p[(i + 1) % len(p)])
        return 0.5 * abs(np.dot(s, self.normale()))

    def aire_nette(self) -> float:
        return self.aire() - self.aire_trous


@dataclass
class Geometrie:
    faces: list[Face]
    fixation: str
    appui: str
    colonnes_vis: tuple[float, ...]
    valeurs: dict = field(default_factory=dict)
    erreurs_evidements: list[str] = field(default_factory=list)
    notes_evidements: list[str] = field(default_factory=list)

    def face(self, ident: str) -> Face:
        return next(f for f in self.faces if f.id == ident)


class ConceptJSON:
    """Concept chargé depuis le format déclaratif ; s'utilise comme EquerreU dans `evaluation.evaluer`."""

    def __init__(self, definition: dict, valeurs: dict, cdc: CahierDesCharges):
        self.definition = definition
        self.valeurs = dict(valeurs)
        self.cdc = cdc
        self._geo: Geometrie | None = None
        self._erreur_construction: str | None = None
        try:
            self._geo = self._construire()
        except Exception as e:  # une expression fausse ou une face mal formée = concept invalide
            self._erreur_construction = f"construction impossible : {e}"

    @classmethod
    def charger(cls, chemin: str | Path, valeurs: dict, cdc: CahierDesCharges) -> "ConceptJSON":
        return cls(json.loads(Path(chemin).read_text(encoding="utf-8")), valeurs, cdc)

    # ---------- construction ----------
    def _env(self) -> dict:
        ves = self.cdc.procede.ves_disponibles
        return {"min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan, "atan2": math.atan2, "radians": math.radians, "degrees": math.degrees,
                "aile_min": lambda t: vé_et_aile(t, ves)[1], "X_FACE": X_FACE,
                "X_CHARGE": X_FACE + self.cdc.chargements[0].distance_encastrement, "R_APPUI": 25.0,
                "Y_MAX": self.cdc.encombrement.y_max, "X_MAX": self.cdc.encombrement.x_max}

    def _construire(self) -> Geometrie:
        d = self.definition
        env = self._env()
        for nom, spec in d["variables"].items():
            if spec["type"] in ("retour", "evidement"):
                continue
            env[nom] = float(self.valeurs[nom])
        for nom, spec in d["variables"].items():  # retours et évidements dépendent de l'épaisseur de leur tôle
            if spec["type"] in ("retour", "evidement"):
                env[nom] = float(self.valeurs.get(nom, 0.0))
        for nom, expr in d.get("derivees", {}).items():
            env[nom] = evaluer_expr(expr, env)
        self.env = env
        faces = []
        declarees: dict[str, set[str]] = {}
        for tole in d["toles"]:
            t = evaluer_expr(tole["epaisseur"], env)
            declarees[tole["nom"]] = set(tole["faces"]) | {
                r.get("nom", f"retour_{r['face']}_{k}") for k, r in enumerate(tole.get("retours", []))}
            locales = {}
            for nom, pts in tole["faces"].items():
                if isinstance(pts, dict):  # face optionnelle : {"points": [...], "si": "expression > 0"}
                    if evaluer_expr(pts.get("si", 1), env) <= 1e-9:
                        continue
                    pts = pts["points"]
                p = np.array([[evaluer_expr(c, env) for c in pt] for pt in pts])
                locales[nom] = Face(tole["nom"], nom, t, p)
            for k, r in enumerate(tole.get("retours", [])):
                h = evaluer_expr(r["hauteur"], env)
                if h <= 1e-9:
                    continue
                base = locales[r["face"]]
                i, j = r["arete"]
                a, b = base.points[i], base.points[j]
                v = np.array([evaluer_expr(c, env) for c in r["direction"]])
                v = v / np.linalg.norm(v)
                nom = r.get("nom", f"retour_{r['face']}_{k}")
                locales[nom] = Face(tole["nom"], nom, t, np.array([a, b, b + h * v, a + h * v]))
            faces += locales.values()
        fix = d["fixation"]
        ecart = env[fix["colonnes_vis"]] if isinstance(fix.get("colonnes_vis"), str) else float(fix.get("colonnes_vis", 0))
        colonnes = (-ecart, ecart) if ecart > 0 else (0.0,)
        g = Geometrie(faces, fix["face"], d["appui"]["face"], colonnes, dict(self.valeurs))
        self._evidements(g, env, declarees)
        return g

    def _evidements(self, g: Geometrie, env: dict, declarees: dict[str, set[str]]) -> None:
        """Perce les faces : les zones protégées dépendent de tout l'assemblage, d'où ce calcul en dernier."""
        par_face: dict[str, list[dict]] = {}
        for tole in self.definition["toles"]:
            for k, s in enumerate(tole.get("evidements", []), start=1):
                face = s.get("face")
                if face not in declarees[tole["nom"]]:
                    raise ValueError(f"évidement {k} de la tôle {tole['nom']} : face « {face} » inconnue")
                if evaluer_expr(s.get("si", 1), env) <= 1e-9:
                    continue
                par_face.setdefault(f"{tole['nom']}.{face}", []).append(_evaluer_evidement(s, env))
        if not par_face:
            return
        ctx = contexte_evidements(g, self.cdc, env["X_CHARGE"])
        for f in g.faces:
            if f.id not in par_face:
                continue
            r = ev.construire(f, par_face[f.id], ctx)
            g.erreurs_evidements += r.erreurs
            g.notes_evidements += r.notes
            f.trous, f.aire_trous = r.trous, r.aire
            f.rayons_trous = list(r.rayons)  # rayon des arrondis (les coins rentrants ne concentrent pas)

    # ---------- vérifications ----------
    def verifier(self, cdc: CahierDesCharges | None = None) -> list[str]:
        cdc = cdc or self.cdc
        if self._erreur_construction:
            return [self._erreur_construction]
        g = self._geo
        e = list(g.erreurs_evidements)
        toles = {f.tole for f in g.faces}
        # une face doublée contre la face de fixation est physiquement devant elle : son plan moyen est
        # celui du doublage, il peut donc être à moins d'une demi-épaisseur propre de la face du socle
        x_fix = next((f.points[:, 0].min() for f in g.faces if f.id == g.fixation), X_FACE)
        if len(toles) > cdc.procede.nombre_pieces_max:
            e.append(f"{len(toles)} tôles pour {cdc.procede.nombre_pieces_max} autorisées")
        for f in g.faces:
            if f.t not in cdc.procede.epaisseurs_disponibles:
                e.append(f"{f.id} : épaisseur {f.t} absente de l'atelier")
            if f.aire() < 1.0:
                e.append(f"{f.id} : face dégénérée")
            ecart_plan = np.abs((f.points - f.points[0]) @ f.normale()).max()
            if ecart_plan > 1e-3:
                e.append(f"{f.id} : face non plane ({ecart_plan:.2f} mm)")
            pts = f.points
            if (np.abs(pts[:, 1]) + f.t / 2 > cdc.encombrement.y_max + 1e-6).any():
                e.append(f"{f.id} : trop large, touche les nervures du socle")
            if (pts[:, 2] < cdc.encombrement.z_min + DEGAGEMENT_SEMELLE - 1e-6).any():
                e.append(f"{f.id} : à moins de {DEGAGEMENT_SEMELLE:.0f} mm de la semelle")
            if (pts[:, 0] > cdc.encombrement.x_max + 1e-6).any():
                e.append(f"{f.id} : dépasse le bout de la semelle")
            if (pts[:, 0] < min(X_FACE + f.t / 2, x_fix) - 1e-6).any():
                e.append(f"{f.id} : rentre dans le socle")
        e += self._verifier_plis(cdc)
        try:
            fix = g.face(g.fixation)
            if abs(abs(fix.normale()[0]) - 1) > 1e-6:
                e.append("la face de fixation doit être plaquée sur la face du socle (plan x = constante)")
            elif not self.trous_utilises(cdc):
                e.append("aucun trou de la grille n'est utilisable sur la face de fixation")
        except StopIteration:
            e.append(f"face de fixation {g.fixation} introuvable")
        try:
            ap = g.face(g.appui)
            xc = self.env["X_CHARGE"]
            if abs(abs(ap.normale()[2]) - 1) > 1e-6:
                e.append("la face d'appui doit être horizontale")
            elif not _dans_polygone(ap.points[:, :2], (xc, 0.0)):
                e.append("le centre de l'appui (x = 320, y = 0) n'est pas sur la face d'appui")
        except StopIteration:
            e.append(f"face d'appui {g.appui} introuvable")
        if not e:
            e += self._verifier_liaisons()
        return e

    def _verifier_liaisons(self) -> list[str]:
        """Chaque soudure déclarée doit exister (arête commune d'au moins 10 mm ou doublage) et l'assemblage
        doit être d'un seul tenant. Découpage OCC seul, sans maillage."""
        g = self._geo
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("liaisons")
        occ = gmsh.model.occ
        entrees = []
        for f in g.faces:
            ids = [occ.addPoint(*p) for p in f.points]
            lignes = [occ.addLine(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
            entrees.append(occ.addPlaneSurface([occ.addCurveLoop(lignes)]))
        carte = _fragmenter(occ, entrees)
        occ.synchronize()
        surfaces, courbes = {}, {}
        for f, morceaux in zip(g.faces, carte):
            surfaces[f.id] = {t for d, t in morceaux if d == 2}
            courbes[f.id] = {abs(c) for _, c in gmsh.model.getBoundary([(2, t) for t in surfaces[f.id]],
                                                                       combined=False, oriented=False)}
            for t in surfaces[f.id]:  # arête d'une autre face posée à l'intérieur de celle-ci (jonction en T)
                courbes[f.id] |= {c for d, c in gmsh.model.mesh.getEmbedded(2, t) if d == 1}
        longueur = {c: occ.getMass(1, c) for cs in courbes.values() for c in cs}
        gmsh.finalize()

        def liees(a, b):
            return bool(surfaces[a] & surfaces[b]) or sum(longueur[c] for c in courbes[a] & courbes[b]) >= 10.0

        e = []
        ids = [f.id for f in g.faces]
        for a, b in self.definition.get("soudures", []):
            if a not in surfaces or b not in surfaces:
                manque = a if a not in surfaces else b
                e.append(f"soudure {a}/{b} : la face {manque} n'existe pas pour ces valeurs")
            elif not liees(a, b):
                e.append(f"soudure {a}/{b} : les deux faces ne se touchent pas (ni arête commune, ni doublage)")
        groupe = {ids[0]}
        while True:
            ajout = {b for b in ids if b not in groupe and any(liees(a, b) for a in groupe)}
            if not ajout:
                break
            groupe |= ajout
        isolees = [i for i in ids if i not in groupe]
        if isolees:
            e.append(f"assemblage en plusieurs morceaux : {', '.join(isolees)} non reliées à {ids[0]}")
        return e

    def _verifier_plis(self, cdc) -> list[str]:
        e = []
        g = self._geo
        for i, f1 in enumerate(g.faces):
            for f2 in g.faces[i + 1:]:
                if f1.tole != f2.tole:
                    continue
                arete = _arete_commune(f1.points, f2.points)
                if arete is None:
                    continue
                a, b = arete
                u = (b - a) / np.linalg.norm(b - a)
                d1, d2 = _direction_interieure(f1, a, u), _direction_interieure(f2, a, u)
                angle = math.degrees(math.acos(np.clip(np.dot(d1, d2), -1, 1)))
                if angle > 179.0:
                    continue  # faces dans le prolongement : pas un pli
                if angle < ANGLE_PLI_MIN:
                    e.append(f"pli {f1.id}/{f2.id} : angle intérieur {angle:.0f}° < {ANGLE_PLI_MIN:.0f}°")
                _, aile = vé_et_aile(f1.t, cdc.procede.ves_disponibles)
                for f, d in ((f1, d1), (f2, d2)):
                    portee = ((f.points - a) @ d).max()
                    if portee < aile - 1e-6:
                        e.append(f"pli {f1.id}/{f2.id} : aile {f.nom} de {portee:.1f} mm < aile mini {aile:.1f} mm")
        return e

    def trous_utilises(self, cdc: CahierDesCharges | None = None) -> list[tuple[float, float]]:
        return _trous_utilises(self._geo, cdc or self.cdc)

    def masse(self, rho: float) -> float:
        return sum(f.aire_nette() * f.t for f in self._geo.faces) * rho * 1e-9

    # ---------- maillage et modèle ----------
    def mailler(self, taille: float = 6.0):
        g = self._geo
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("concept")
        occ = gmsh.model.occ
        entrees = [surface_percee(occ, f) for f in g.faces]
        carte = _fragmenter(occ, entrees)
        occ.synchronize()
        couverture: dict[int, list[Face]] = {}
        for f, morceaux in zip(g.faces, carte):
            for dim, tag in morceaux:
                if dim == 2:
                    couverture.setdefault(tag, []).append(f)
        if any(f.trous for f in g.faces):
            _raffiner_trous(g, taille)
        else:  # sans évidement : maillage uniforme, identique aux versions précédentes
            gmsh.option.setNumber("Mesh.MeshSizeMax", 2 * taille)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 2 * taille)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        for essai in range(2):
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)
            tags, coords, _ = gmsh.model.mesh.getNodes()
            noeuds = {int(t): tuple(coords[3 * k : 3 * k + 3]) for k, t in enumerate(tags)}
            elements, epaisseurs = {}, {}
            self.faces_elements: dict[int, list[str]] = {}  # élément -> faces qu'il représente (2 pour un doublage)
            for tag, faces_couvrantes in couverture.items():
                t = sum({f.tole: f.t for f in faces_couvrantes}.values())  # doublage : épaisseurs additionnées
                types, etags, enodes = gmsh.model.mesh.getElements(2, tag)
                for ty, et, en in zip(types, etags, enodes):
                    if ty != 16:
                        raise RuntimeError(f"élément de type {ty} inattendu")
                    for k, e in enumerate(et):
                        elements[int(e)] = [int(n) for n in en[8 * k : 8 * k + 8]]
                        epaisseurs[int(e)] = t
                        self.faces_elements[int(e)] = [f.id for f in faces_couvrantes]
            mauvais = elements_retournes(noeuds, elements)
            if not mauvais:
                break
            if essai == 0:  # second essai : autre algorithme de surface et lissage renforcé
                gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm", 8)
                gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.finalize()
        if mauvais:
            faces = sorted({f for e in mauvais for f in self.faces_elements.get(e, [])})
            raise RuntimeError(f"maillage invalide : {len(mauvais)} éléments retournés ({', '.join(faces)})")
        utilises = {n for c in elements.values() for n in c}
        return {n: c for n, c in noeuds.items() if n in utilises}, elements, epaisseurs

    def modele(self, cdc: CahierDesCharges, taille: float = 6.0) -> tuple[ModeleCoque, dict]:
        g = self._geo
        noeuds, elements, epaisseurs = self.mailler(taille)
        if not _connexe(elements):
            raise RuntimeError("l'assemblage n'est pas d'un seul tenant (une tôle flotte)")
        self._controler_soudures_maillees(elements)
        fix, ap = g.face(g.fixation), g.face(g.appui)
        xd, zc = fix.points[0, 0], ap.points[0, 2]
        mat = cdc.materiau
        materiau = Materiau(mat.nom, mat.E, mat.nu, rho=mat.rho * 1e-12, Re=mat.Re)
        trous = self.trous_utilises(cdc)
        encastres = {n for n, (x, y, z) in noeuds.items()
                     if abs(x - xd) < 1e-6 and any(np.hypot(y - ty, z - tz) <= RAYON_APPUI_VIS for ty, tz in trous)}
        ch = cdc.chargements[0]
        xc = self.env["X_CHARGE"]
        disque = [n for n, (x, y, z) in noeuds.items() if abs(z - zc) < 1e-6 and np.hypot(x - xc, y) <= 25.0]
        f = np.array(ch.direction) * ch.force / len(disque)
        dos = {n for n, (x, y, z) in noeuds.items() if abs(x - xd) < 1e-6} - encastres
        t_dos = max(epaisseurs[e] for e, c in elements.items() if all(abs(noeuds[n][0] - xd) < 1e-6 for n in c))
        info = {"trous": trous, "dos": dos, "x_charge": xc, "noeuds_charge": disque, "x_dos": xd,
                "t_dos": t_dos, "z_charge": zc, "t_charge": ap.t}
        modele = ModeleCoque(noeuds, elements, ap.t, materiau, encastres, {n: tuple(f) for n in disque},
                             epaisseurs=epaisseurs)
        return modele, info

    def _controler_soudures_maillees(self, elements: dict[int, list[int]]) -> None:
        """Chaque soudure déclarée doit partager des noeuds dans le maillage : sinon le calcul porterait sur
        une pièce dont une liaison a disparu (mécanisme), sans autre symptôme qu'une flèche absurde."""
        noeuds_face: dict[str, set[int]] = {}
        for e, faces in self.faces_elements.items():
            for f in faces:
                noeuds_face.setdefault(f, set()).update(elements[e])
        for a, b in self.definition.get("soudures", []):
            if a in noeuds_face and b in noeuds_face and len(noeuds_face[a] & noeuds_face[b]) < 3:
                raise RuntimeError(f"soudure {a}/{b} absente du maillage (aucun noeud commun)")

    def contexte_evidements(self) -> ev.Contexte:
        """Ce qu'il faut savoir de l'assemblage pour placer une découpe (zones protégées)."""
        return contexte_evidements(self._geo, self.cdc, self.env["X_CHARGE"])

    def faces_proches(self, p, tol: float = 1.0) -> list[str]:
        """Faces qui passent par le point p (à une demi-épaisseur près) : sert à nommer une zone du modèle."""
        p = np.asarray(p, float)
        res = []
        for f in self._geo.faces:
            n = f.normale()
            if abs(np.dot(p - f.points[0], n)) > f.t + tol:
                continue
            q = p - np.dot(p - f.points[0], n) * n  # projection dans le plan de la face
            axes = np.linalg.svd(f.points - f.points.mean(0))[2][:2]
            poly = (f.points - f.points[0]) @ axes.T
            loc = (q - f.points[0]) @ axes.T
            if _dans_polygone(poly, loc) or _distance_bords(poly, loc) <= f.t + tol:
                res.append(f.id)
        return res

    def parametres(self) -> dict:
        return dict(self.valeurs)

    def geometrie(self) -> Geometrie:
        return self._geo


# ---------- évidements ----------
TAILLE_TROU_RELATIVE = 1 / 3  # taille de maille au bord d'un évidement / rayon de courbure du bord
TAILLE_TROU_MIN = 1.5  # mm


def _evaluer_evidement(s: dict, env: dict) -> dict:
    ty = s.get("type")
    e = {"type": ty}
    if ty == "decalage":
        e["marge"] = evaluer_expr(s.get("marge", 0), env)
        e["rayon"] = evaluer_expr(s["rayon"], env) if "rayon" in s else None
        e["largeur_nervure"] = evaluer_expr(s["largeur_nervure"], env) if "largeur_nervure" in s else None
        e["nervures"] = [[[evaluer_expr(c, env) for c in bout] for bout in n] for n in s.get("nervures", [])]
    elif ty == "cercle":
        e["centre"] = [evaluer_expr(c, env) for c in s["centre"]]
        e["diametre"] = evaluer_expr(s.get("diametre", 0), env)
    elif ty == "polygone":
        e["points"] = [[evaluer_expr(c, env) for c in pt] for pt in s["points"]]
        e["rayon"] = evaluer_expr(s["rayon"], env) if "rayon" in s else None
    return e


def contexte_evidements(g: Geometrie, cdc: CahierDesCharges, x_charge: float) -> ev.Contexte:
    ap = next((f for f in g.faces if f.id == g.appui), None)
    return ev.Contexte(g.faces, g.fixation, g.appui, _trous_utilises(g, cdc),
                       np.array([x_charge, 0.0, ap.points[0, 2] if ap is not None else 0.0]),
                       lambda t: _distance_pli(t, cdc.procede.ves_disponibles))


def _distance_pli(t: float, ves: list[float]) -> float:
    """Distance protégée de part et d'autre d'une ligne de pli : 2,5 t + rayon intérieur (≈ 0,16 x vé)."""
    v, _ = vé_et_aile(t, ves)
    return 2.5 * t + 0.16 * v


def _trous_utilises(g: Geometrie, cdc: CahierDesCharges) -> list[tuple[float, float]]:
    fix = g.face(g.fixation)
    poly = fix.points[:, 1:]
    return [(h.y, h.z) for h in cdc.grille_fixation.trous
            if any(abs(h.y - c) < 1e-6 for c in g.colonnes_vis)
            and _dans_polygone(poly, (h.y, h.z)) and _distance_bords(poly, (h.y, h.z)) >= MARGE_TROU]


def _fragmenter(occ, entrees: list[int]) -> list[list[tuple[int, int]]]:
    """Découpage commun des surfaces (Gmsh saute le découpage d'une surface seule : on la garde telle quelle)."""
    if len(entrees) == 1:
        return [[(2, entrees[0])]]
    return occ.fragment([(2, s) for s in entrees], [])[1]


def courbes_contour(occ, anneau: np.ndarray) -> list[int]:
    """Courbes OCC d'un contour d'évidement : droites pour les bords droits, splines pour les arrondis."""
    courbes = []
    morceaux = ev.chaines_contour(anneau)
    if morceaux[0][0] == "spline_fermee":
        ids = [occ.addPoint(*p) for p in morceaux[0][1]]
        return [occ.addSpline(ids + [ids[0]])]
    premier = None
    precedent = None
    for k, (nature, pts) in enumerate(morceaux):
        debut = precedent if precedent is not None else occ.addPoint(*pts[0])
        premier = debut if premier is None else premier
        fin = premier if k == len(morceaux) - 1 else occ.addPoint(*pts[-1])
        if nature == "ligne":
            courbes.append(occ.addLine(debut, fin))
        else:
            milieu = [occ.addPoint(*p) for p in pts[1:-1]]
            courbes.append(occ.addSpline([debut, *milieu, fin]))
        precedent = fin
    return courbes


def surface_percee(occ, f: Face, points: np.ndarray | None = None, decalage=0.0) -> int:
    """Surface plane OCC d'une face, avec ses évidements (points éventuellement décalés pour l'export)."""
    p = f.points if points is None else points
    ids = [occ.addPoint(*q) for q in p]
    lignes = [occ.addLine(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    boucles = [occ.addCurveLoop(lignes)]
    rep = ev.Repere.de(f.points, f.normale())
    sens = np.sign(_aire_signee(rep.vers_2d(f.points)))
    for trou in f.trous:
        trou = np.asarray(trou)
        if np.sign(_aire_signee(rep.vers_2d(trou))) != sens:
            trou = trou[::-1]  # Gmsh veut les trous dans le même sens que le contour (constaté : sinon OCC
            # ajoute l'aire du trou au lieu de la retrancher, et les jonctions en T se perdent)
        boucles.append(occ.addCurveLoop(courbes_contour(occ, trou + decalage)))
    return occ.addPlaneSurface(boucles)


def _aire_signee(p2: np.ndarray) -> float:
    x, y = p2[:, 0], p2[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _raffiner_trous(g: Geometrie, taille: float) -> None:
    """Maille fine au bord des évidements (concentration de contrainte), qui grossit jusqu'à `taille` en
    s'éloignant. Les courbes des trous sont retrouvées par leur position après le découpage OCC. Avec la
    subdivision en quadrangles, la taille finale est la moitié de la taille demandée à Gmsh."""
    trous = [(np.asarray(tr), r) for f in g.faces for tr, r in zip(f.trous, f.rayons_trous)]
    seuils = []
    for c in (tag for _, tag in gmsh.model.getEntities(1)):
        bas, haut = gmsh.model.getParametrizationBounds(1, c)
        pts = np.array(gmsh.model.getValue(1, c, [bas[0] + k * (haut[0] - bas[0]) / 4 for k in range(5)])).reshape(-1, 3)
        for anneau, r in trous:
            if _distance_polyligne(pts, anneau).max() < 0.05 * max(r, 1.0):
                h = min(taille, max(TAILLE_TROU_MIN, TAILLE_TROU_RELATIVE * r))
                seuils.append((c, h))
                break
    champs = []
    for h in sorted({h for _, h in seuils}):
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", [c for c, hc in seuils if hc == h])
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 100)
        seuil = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(seuil, "InField", dist)
        gmsh.model.mesh.field.setNumber(seuil, "SizeMin", 2 * h)
        gmsh.model.mesh.field.setNumber(seuil, "SizeMax", 2 * taille)
        gmsh.model.mesh.field.setNumber(seuil, "DistMin", h)
        gmsh.model.mesh.field.setNumber(seuil, "DistMax", h + 4 * taille)
        champs.append(seuil)
    if champs:
        mini = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mini, "FieldsList", champs)
        gmsh.model.mesh.field.setAsBackgroundMesh(mini)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2 * taille)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 2 * TAILLE_TROU_MIN)


def _distance_polyligne(pts: np.ndarray, anneau: np.ndarray) -> np.ndarray:
    """Distance de chaque point à un contour fermé (segments)."""
    a = anneau
    b = np.roll(anneau, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("pk,sk->ps", pts, ab) - np.einsum("sk,sk->s", a, ab), 0, None)
    t = np.clip(t / np.maximum(np.einsum("sk,sk->s", ab, ab), 1e-12), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None] - proj, axis=2).min(axis=1)


# ---------- outils géométriques ----------
def _dans_polygone(poly: np.ndarray, p) -> bool:
    x, y = p
    dedans = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            dedans = not dedans
    return dedans


def _distance_bords(poly: np.ndarray, p) -> float:
    p = np.asarray(p, float)
    d = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        s = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        d = min(d, float(np.linalg.norm(p - (a + s * ab))))
    return d


def _arete_commune(p1: np.ndarray, p2: np.ndarray, tol: float = 1e-6):
    """Segment commun (au moins une longueur > 1 mm) entre une arête de p1 et une arête de p2."""
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
            s = sorted([0.0, L])
            t = sorted([np.dot(c - a, u), np.dot(d - a, u)])
            lo, hi = max(s[0], t[0]), min(s[1], t[1])
            if hi - lo > 1.0:
                return a + lo * u, a + hi * u
    return None


def _direction_interieure(f: Face, a: np.ndarray, u: np.ndarray) -> np.ndarray:
    d = np.cross(f.normale(), u)
    centre = f.points.mean(axis=0)
    return d if np.dot(centre - a, d) > 0 else -d


def elements_retournes(noeuds: dict, elements: dict[int, list[int]]) -> list[int]:
    """Quadrangles non convexes ou retournés : à un coin, le produit vectoriel des deux arêtes n'a pas le sens
    de la normale de l'élément (CalculiX refuserait le calcul : jacobien négatif)."""
    if not elements:
        return []
    ids = list(elements)
    p = np.array([[noeuds[n] for n in elements[e][:4]] for e in ids])  # (n, 4, 3)
    normale = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    mauvais = np.zeros(len(ids), bool)
    for k in range(4):
        a, b, c = p[:, k], p[:, (k + 1) % 4], p[:, (k + 3) % 4]
        mauvais |= np.einsum("ij,ij->i", np.cross(b - a, c - a), normale) <= 0
    return [ids[i] for i in np.flatnonzero(mauvais)]


def _connexe(elements: dict[int, list[int]]) -> bool:
    parent: dict[int, int] = {}

    def racine(n):
        while parent.setdefault(n, n) != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for conn in elements.values():
        r0 = racine(conn[0])
        for n in conn[1:]:
            parent[racine(n)] = r0
    return len({racine(n) for n in parent}) == 1
