"""Tests des évidements (sans CalculiX) : zones protégées, règles de fabrication, îlots, masse, variable
« evidement », maillage commun des soudures avec une face percée.

Lancer :  python tests/tester_evidements.py
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from gabarit.cahier_des_charges import CahierDesCharges  # noqa: E402
from gabarit.concepts import evidements as ev  # noqa: E402
from gabarit.concepts.format import ConceptJSON, _distance_pli  # noqa: E402
from gabarit.etudes.concept_json import proposer_valeurs  # noqa: E402

CDC = CahierDesCharges.charger(RACINE / "cahiers" / "sae51_equerre.json")
BASE = json.loads((RACINE / "concepts" / "sae51_L_V.json").read_text(encoding="utf-8"))
VALEURS = json.loads((RACINE / "resultats" / "sae51_L_V" / "synthese.json").read_text(encoding="utf-8"))["valeurs"]
resultats: list[tuple[str, bool]] = []


def verifier(nom: str, ok: bool) -> None:
    resultats.append((nom, bool(ok)))


def concept(ev_L=(), ev_V=(), variables=None, valeurs=None) -> ConceptJSON:
    d = copy.deepcopy(BASE)
    d["toles"][0]["evidements"] = list(ev_L)
    d["toles"][1]["evidements"] = list(ev_V)
    d["variables"].update(variables or {})
    return ConceptJSON(d, {**VALEURS, **(valeurs or {})}, CDC)


def trous_2d(f) -> list[Polygon]:
    rep = ev.Repere.de(f.points, f.normale())
    return [Polygon(rep.vers_2d(t)) for t in f.trous], rep


# 1. poche « decalage » sur le dessus : hors des zones protégées
c = concept([{"face": "dessus", "type": "decalage", "marge": 15}])
f = c.geometrie().face("L.dessus")
polys, rep = trous_2d(f)
verifier("poche du dessus construite, sans erreur", c.verifier(CDC) == [] and len(polys) == 1)
charge = Point(rep.vers_2d([c.env["X_CHARGE"], 0, f.points[0, 2]]))
verifier("à 35 mm au moins du centre de l'appui", min(p.distance(charge) for p in polys) >= ev.RAYON_APPUI_PROTEGE - 0.01)
x_diag = VALEURS["x_diag"]
soudure = LineString(rep.vers_2d([[x_diag, -40, f.points[0, 2]], [x_diag, 40, f.points[0, 2]]]))
verifier("à 15 mm au moins de la soudure de la diagonale", min(p.distance(soudure) for p in polys) >= 15 - 0.01)
pli = LineString(rep.vers_2d(f.points[:2]))  # arête 0 du dessus = pli avec le dos
verifier("à 2,5 t + 0,16 vé au moins du pli avec le dos",
         min(p.distance(pli) for p in polys) >= _distance_pli(5.0, CDC.procede.ves_disponibles) - 0.01)
bord = Polygon(rep.vers_2d(f.points)).exterior
verifier("bande de 15 mm au moins sur les bords", min(p.distance(bord) for p in polys) >= 15 - 0.01)
verifier("masse : aire découpée x épaisseur x densité",
         abs(concept().masse(7850) - c.masse(7850) - f.aire_trous * 5.0 * 7850e-9) < 1e-9)

# 2. nervure : la poche est partagée en deux
c = concept([{"face": "dessus", "type": "decalage", "marge": 15, "nervures": [[[0, 0.5], [2, 0.5]]]}])
polys, _ = trous_2d(c.geometrie().face("L.dessus"))
verifier("une nervure partage la poche en deux", len(polys) == 2 and polys[0].distance(polys[1]) >= 10 - 0.01)

# 3. règles : marge trop faible, trou dans un doublage, trou trop près du bord
c = concept([{"face": "dessus", "type": "decalage", "marge": 6}])
verifier("marge < bande minimale refusée", any("bande minimale" in e for e in c.verifier(CDC)))
c = concept(ev_V=[{"face": "jambe", "type": "cercle", "centre": ["xd", 0, "z_pied + h_jambe / 2"], "diametre": 20}])
verifier("trou dans le doublage jambe / dos refusé", any("zone protégée" in e for e in c.verifier(CDC)))
c = concept([{"face": "dessus", "type": "cercle", "centre": [200, "b / 2 - 12", "z_dessus"], "diametre": 16}])
verifier("trou à moins de 10 mm du bord refusé", any("du bord" in e for e in c.verifier(CDC)))
c = concept([{"face": "dessus", "type": "cercle", "centre": [200, 0, "z_dessus"], "diametre": 30},
             {"face": "dessus", "type": "cercle", "centre": [232, 0, "z_dessus"], "diametre": 30}])
verifier("deux trous trop proches refusés", any("l'un de l'autre" in e for e in c.verifier(CDC)))
c = concept([{"face": "dessus", "type": "cercle", "centre": [200, 0, "z_dessus"], "diametre": 30}])
verifier("trou bien placé accepté", c.verifier(CDC) == [] and len(c.geometrie().face("L.dessus").trous) == 1)
c = concept([{"face": "dessous", "type": "decalage", "marge": 15}])
verifier("face inconnue signalée", c.verifier(CDC) and "inconnue" in c.verifier(CDC)[0])

# 4. îlots : une poche sur le dos entoure les vis ; chaque vis reste reliée au reste de la tôle
d_seul = copy.deepcopy(BASE)
d_seul["toles"] = [d_seul["toles"][0]]
d_seul["toles"][0]["evidements"] = [{"face": "dos", "type": "decalage", "marge": 12}]
d_seul["soudures"] = []
c = ConceptJSON(d_seul, {**VALEURS, "ecart_vis": 40, "h_ret_dos": 0.0}, CDC)
f = c.geometrie().face("L.dos")
polys, rep = trous_2d(f)
matiere = Polygon(rep.vers_2d(f.points)).difference(unary_union(polys)) if polys else None
verifier("poche du dos créée autour des vis", len(polys) >= 1)
verifier("aucun îlot : la tôle percée reste d'un seul tenant",
         matiere is not None and matiere.geom_type == "Polygon" and all(not p.interiors for p in polys))
vis = [Point(rep.vers_2d([f.points[0, 0], y, z])) for y, z in c.trous_utilises(CDC)]
verifier("12 mm de matière au moins autour de chaque vis",
         bool(vis) and min(p.distance(v) for p in polys for v in vis) >= ev.MARGE_VIS - 0.01)

# 5. variable « evidement » : 0 (absente) ou entre la bande minimale et le maximum
var = {"m_diag": {"type": "evidement", "tole": "V", "max": 40}}
d = copy.deepcopy(BASE)
d["variables"].update(var)


class Tirage:
    def __init__(self, actif, u):
        self.actif, self.u = actif, u

    def suggest_categorical(self, nom, choix):
        if nom.endswith("_actif"):
            return self.actif
        return choix[0]

    def suggest_float(self, nom, bas, haut):
        return bas + self.u * (haut - bas)


v0 = proposer_valeurs(d, CDC, Tirage(False, 0.5))
v1 = proposer_valeurs(d, CDC, Tirage(True, 0.0))
v2 = proposer_valeurs(d, CDC, Tirage(True, 1.0))
t_V = v1["t_V"]
verifier("évidement absent : marge 0", v0["m_diag"] == 0.0)
verifier("évidement présent : marge entre la bande minimale et le maximum",
         math.isclose(v1["m_diag"], ev.ligament_min(t_V)) and math.isclose(v2["m_diag"], 40.0))

# 6. maillage : la soudure de la diagonale sous un dessus percé reste maillée en commun (piège Gmsh du sens
#    des contours de trous : sinon OCC ajoute l'aire du trou et la jonction en T disparaît)
c = concept([{"face": "dessus", "type": "decalage", "marge": 15}], [{"face": "diag", "type": "decalage", "marge": 15}])
noeuds, elements, _ = c.mailler(8.0)
faces_el = c.faces_elements
n_dessus = {n for e, fs in faces_el.items() if "L.dessus" in fs for n in elements[e]}
n_diag = {n for e, fs in faces_el.items() if "V.diag" in fs for n in elements[e]}
verifier("soudure diagonale / dessus percé : noeuds communs", len(n_dessus & n_diag) >= 10)
aire_maillee = 0.0
for e, fs in faces_el.items():
    if fs == ["L.dessus"]:
        p = np.array([noeuds[n] for n in elements[e][:4]])
        aire_maillee += 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[1]))
f = c.geometrie().face("L.dessus")
verifier("aire maillée du dessus = aire percée (à 1 % près)", abs(aire_maillee - f.aire_nette()) < 0.01 * f.aire_nette())
try:
    c.modele(CDC, 8.0)
    verifier("contrôle des soudures maillées passé", True)
except RuntimeError:
    verifier("contrôle des soudures maillées passé", False)

# 7. contours pour la CAO : droites pour les bords droits, splines pour les arrondis
morceaux = ev.chaines_contour(np.asarray(c.geometrie().face("L.dessus").trous[0]))
natures = {m[0] for m in morceaux}
verifier("contour de poche : droites et arrondis séparés", natures == {"ligne", "spline"})
cercle = np.array([[10 * math.cos(a), 10 * math.sin(a), 0] for a in np.linspace(0, 2 * math.pi, 64, endpoint=False)])
verifier("cercle : une seule spline fermée", ev.chaines_contour(cercle)[0][0] == "spline_fermee")

for nom, ok in resultats:
    print(f"{'OK ' if ok else 'ÉCHEC'} {nom}")
echecs = sum(not ok for _, ok in resultats)
print(f"\n{len(resultats) - echecs}/{len(resultats)} tests réussis")
sys.exit(1 if echecs else 0)
