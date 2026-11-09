"""Optimisation topologique des tôles (ESO) et poches d'allègement qui en découlent.

Méthode ESO (Evolutionary Structural Optimization, Xie et Steven, 1993), en version « tout ou rien » sur le
maillage coque du concept, aux valeurs de ses variables :
1. zone évidable de chaque face : les éléments dont tout le quadrilatère reste hors des zones protégées
   (plis, soudures, doublages, vis, appui : mêmes règles que les évidements) et à plus d'une bande minimale
   max(2 t, 10 mm) du bord ;
2. à chaque itération : calcul statique avec contact, puis on retire, parmi les éléments évidables restants,
   ceux dont la contrainte de peau filtrée est la plus faible, jusqu'à `pas` (5 %) du volume évidable. Le
   filtre (moyenne pondérée par la distance sur un rayon de 2 mailles, dans la face) évite les damiers et
   rend le résultat peu dépendant du maillage ;
3. on va jusqu'à `part_max` sans s'arrêter sur la contrainte : le bord en escalier des éléments retirés crée
   des pics de contrainte artificiels (constaté : +10 % dès 5 % retirés sur L + V), si bien que le kg/kN des
   itérations ne dit rien de celui des vraies poches. L'ESO ne sert qu'à classer la matière, de la moins utile
   à la plus utile ; ce sont les candidats recalculés (plus bas) qui décident.
Limite connue, mesurée sur L + V à son optimum : la matière peu contrainte n'est pas forcément inutile. Le milieu
du dessus près du dos travaille à 20-30 % de la contrainte utile, mais il transmet la traction du dessus au dos ;
une poche de 45 cm² à cet endroit (4 % de la masse) renvoie cette traction dans les bandes latérales, déplace le
point critique de l'avant du dessus (sous l'appui) à son coin arrière près du pli, et fait monter la contrainte
utile de 138 à 196 MPa : kg/kN x1,36. Le critère de l'ESO (contrainte de l'élément lui-même) ignore cet effet sur
le pic, d'où la règle : ce sont les candidats recalculés qui décident, et chacun dit où est passé le point
critique.
Le flambement n'est pas suivi pendant les itérations (trop coûteux) : une face qui participe au premier mode
de flambement n'est pas évidée quand la marge au flambement est faible (une poche l'affaiblirait encore), et
chaque candidat est vérifié au flambement.

Les zones retirées deviennent des poches fabricables : union des éléments retirés, fermeture puis ouverture
morphologiques au rayon par défaut max(2 t, 10 mm) (coins arrondis, parties trop étroites supprimées), zones
protégées et bande de bord respectées, îlots reliés au bord, morceaux de moins de 500 mm² ignorés. Chaque
niveau candidat donne un concept « figé » (variables fixées à leurs valeurs, poches en « polygone ») recalculé
par la chaîne complète (maillage raffiné au bord des trous, flambement confirmé sur deux maillages) : c'est ce
calcul, pas l'ESO, qui décide.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely import prepared
from shapely.geometry import Polygon
from shapely.ops import unary_union

from gabarit.analyse.champ import champ_elements
from gabarit.analyse.compte_rendu import FLAMBEMENT_MIN, FT_RD_M10_88, carte_face
from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts import evidements as ev
from gabarit.concepts.format import ConceptJSON, _connexe
from gabarit.evaluation import _contact, evaluer, pic_utile
from gabarit.fem.calculix import ModeleCoque, lancer

PAS = 0.05  # part du volume évidable retirée à chaque itération
PART_MAX = 0.8
NIVEAUX = (0.15, 0.3, 0.45, 0.6)  # parts retirées par l'ESO dont on tire des poches à recalculer
MARGE_FLAMBEMENT_EVIDEMENT = 1.5  # sous cette marge, les faces du premier mode de flambement ne sont pas évidées
PARTICIPATION_FLAMBEMENT = 0.5
RETRAIT_EXPORT = 0.1  # mm : marge de sécurité des poches exportées (arrondis des coordonnées)
ITERATIONS_CONTACT = 2


@dataclass
class EtapeESO:
    iteration: int
    part_retiree: float  # part du volume évidable retirée
    masse_kg: float
    sigma_utile: float
    effort_tenu_N: float
    kg_par_kN: float
    retires: list[int] = field(default_factory=list)


@dataclass
class Candidat:
    iteration: int
    part_retiree: float
    definition: dict
    poches: dict[str, int]  # face -> nombre de poches
    aire_poches_mm2: float
    masse_kg: float = math.nan
    effort_tenu_N: float = math.nan
    kg_par_kN: float = math.nan
    flambement: float = math.nan
    traction_vis_N: float = math.nan
    sigma_utile: float = math.nan  # MPa, au point critique
    point_critique: list[float] | None = None
    faces_critiques: list[str] = field(default_factory=list)
    faisable: bool = False
    erreurs: list[str] = field(default_factory=list)


@dataclass
class ResultatESO:
    concept: str
    taille: float
    reference: dict  # mesures du concept sans nouvelle poche (même maillage, avec flambement)
    etapes: list[EtapeESO]
    evidable: dict[str, list[int]]
    exclues: dict[str, str] = field(default_factory=dict)  # face -> raison de ne pas l'évider
    candidats: list[Candidat] = field(default_factory=list)
    retenu: Candidat | None = None
    cartes: dict[str, list[str]] = field(default_factory=dict)
    axes: dict[str, list[str]] = field(default_factory=dict)

    def texte(self) -> str:
        r = self.reference
        lignes = [f"## Optimisation topologique (ESO) : {self.concept}",
                  (f"Maille {self.taille:g} mm. Sans nouvelle poche : {r['masse_kg']:.3f} kg, {r['kg_par_kN']:.4f} "
                   f"kg/kN, effort tenu {r['effort_tenu_N'] / 1000:.2f} kN, flambement {r['flambement']:.2f}."),
                  "Faces évidables : " + (", ".join(f"{f} ({len(ids)} éléments)" for f, ids in self.evidable.items())
                                          or "aucune")]
        lignes += [f"{f} non évidée : {raison}" for f, raison in self.exclues.items()]
        lignes.append("Itérations ESO (part du volume évidable retirée : masse, contrainte utile sur le maillage en "
                      "escalier, sans flambement ; indicatif) :")
        for e in self.etapes:
            lignes.append(f"- {100 * e.part_retiree:3.0f} % : {e.masse_kg:.3f} kg, {e.sigma_utile:.0f} MPa, "
                          f"{e.kg_par_kN:.4f} kg/kN")
        lignes.append("Candidats recalculés avec de vraies poches (maillage raffiné, flambement) :")
        for c in self.candidats:
            etat = "faisable" if c.faisable else "NON faisable" + (f" ({'; '.join(c.erreurs[:2])})" if c.erreurs else "")
            lignes.append(f"- ESO {100 * c.part_retiree:.0f} % : {sum(c.poches.values())} poches, "
                          f"{c.aire_poches_mm2 / 100:.0f} cm², {c.masse_kg:.3f} kg, {c.kg_par_kN:.4f} kg/kN, flambement "
                          f"{c.flambement:.2f}, {etat}")
            if math.isfinite(c.masse_kg) and math.isfinite(r.get("sigma_utile", math.nan)):
                lignes.append("  " + explication(r, c))
        if self.retenu is not None:
            gain = 100 * (1 - self.retenu.kg_par_kN / r["kg_par_kN"])
            lignes.append(f"Retenu : ESO {100 * self.retenu.part_retiree:.0f} %, {self.retenu.kg_par_kN:.4f} kg/kN "
                          f"({gain:+.1f} % de mieux que sans poche).")
        else:
            lignes.append("Aucun candidat faisable meilleur que le concept sans poche.")
        if self.cartes:
            lignes.append("Cartes ESO au niveau retenu (# zone protégée, o matière gardée, . matière retirée par l'ESO) :")
            for f, carte in self.cartes.items():
                lignes.append(f"{f} ({' ; '.join(self.axes[f])}) :")
                lignes += [f"  {ligne}" for ligne in carte]
        return "\n".join(lignes)


def zone_evidable(concept: ConceptJSON, modele: ModeleCoque, faces_elements: dict[int, list[str]]) -> dict[str, list[int]]:
    """Éléments évidables de chaque face (un élément de doublage ne l'est jamais : il faudrait percer deux tôles)."""
    geo = concept.geometrie()
    ctx = concept.contexte_evidements()
    par_face: dict[str, list[int]] = {}
    for f in geo.faces:
        rep = ev.Repere.de(f.points, f.normale())
        poly = Polygon(rep.vers_2d(f.points))
        lig = ev.ligament_min(f.t)
        permis = poly.buffer(-lig).difference(ev.zones_protegees(f, ctx))
        for trou in f.trous:  # poches déjà présentes : une bande minimale autour
            permis = permis.difference(Polygon(rep.vers_2d(np.asarray(trou))).buffer(lig))
        if permis.is_empty or permis.area < ev.AIRE_MIN:
            continue
        pp = prepared.prep(permis)
        ids = []
        for e, fs in faces_elements.items():
            if fs != [f.id]:
                continue
            coins = rep.vers_2d(np.array([modele.noeuds[n] for n in modele.elements[e][:4]]))
            if pp.contains(Polygon(coins)):
                ids.append(e)
        if ids:
            par_face[f.id] = ids
    return par_face


def _sous_modele(m0: ModeleCoque, retires: set[int], appuis: set[int]) -> ModeleCoque:
    elements = {e: c for e, c in m0.elements.items() if e not in retires}
    utilises = {n for c in elements.values() for n in c}
    return ModeleCoque({n: p for n, p in m0.noeuds.items() if n in utilises}, elements, m0.epaisseur, m0.materiau,
                       m0.encastres & utilises, {n: f for n, f in m0.forces.items() if n in utilises},
                       appuis & utilises, {e: t for e, t in (m0.epaisseurs or {}).items() if e in elements})


def optimiser_topologie(concept: ConceptJSON, cdc: CahierDesCharges, taille: float = 6.0, pas: float = PAS,
                        part_max: float = PART_MAX, niveaux: tuple[float, ...] = NIVEAUX,
                        dossier: Path | None = None, afficher=print) -> ResultatESO:
    dossier = Path(dossier or tempfile.mkdtemp(prefix="gabarit_eso_"))
    nom = concept.definition.get("nom", "concept")
    F, Re, rho = cdc.chargements[0].force, cdc.materiau.Re, cdc.materiau.rho * 1e-9
    garde: dict = {}
    r0 = evaluer(concept, cdc, taille=taille, dossier=dossier / "reference", flambement=True, garde=garde)
    if not r0.valide:
        raise ValueError(f"concept invalide : {'; '.join(r0.erreurs[:3])}")
    reference = {"masse_kg": r0.masse_kg, "effort_tenu_N": r0.effort_tenu, "kg_par_kN": r0.masse_par_kN,
                 "flambement": r0.facteur_flambement, "traction_vis_N": r0.traction_vis_max,
                 "sigma_utile": r0.sigma_max_utile, "point_critique": _point(r0.point_sigma_utile),
                 "faces_critiques": concept.faces_proches(r0.point_sigma_utile) if r0.point_sigma_utile else []}
    m0, info, faces_el = garde["modele"], garde["info"], garde["faces_elements"]
    epaisseur_face = {f.id: f.t for f in concept.geometrie().faces}
    evidable = zone_evidable(concept, m0, faces_el)
    exclues = {}
    marge = r0.facteur_flambement / FLAMBEMENT_MIN if math.isfinite(r0.facteur_flambement) else 0.0
    if marge < MARGE_FLAMBEMENT_EVIDEMENT and garde.get("flambement") is not None:
        for fid, part in _participation(garde, m0, faces_el, r0.facteur_flambement).items():
            if part >= PARTICIPATION_FLAMBEMENT and fid in evidable:
                exclues[fid] = (f"participe au premier mode de flambement ({100 * part:.0f} %) alors que la marge "
                                f"au flambement n'est que de x{marge:.2f}")
                del evidable[fid]
    tous = sorted(e for ids in evidable.values() for e in ids)
    if not tous:
        return ResultatESO(nom, taille, reference, [], evidable, exclues)
    aire = {e: _aire(m0, e) for e in m0.elements}
    vol = {e: aire[e] * (m0.epaisseurs or {}).get(e, m0.epaisseur) for e in m0.elements}
    v_ev = sum(vol[e] for e in tous)
    # voisins de filtrage : éléments de la même face à moins de 2 mailles
    centres = {e: np.mean([m0.noeuds[n] for n in m0.elements[e][:4]], axis=0) for e in m0.elements}
    rayon_filtre = 2.0 * taille
    voisins: dict[int, list[tuple[int, float]]] = {}
    for fid in evidable:
        ids = [e for e, fs in faces_el.items() if fs == [fid]]
        arbre = cKDTree(np.array([centres[e] for e in ids]))
        for e in evidable[fid]:
            proches = arbre.query_ball_point(centres[e], rayon_filtre)
            voisins[e] = [(ids[k], rayon_filtre - float(np.linalg.norm(centres[ids[k]] - centres[e]))) for k in proches]

    etapes: list[EtapeESO] = []
    retires: set[int] = set()
    appuis = set(m0.appuis_x)
    for it in range(int(part_max / pas) + 2):
        m = _sous_modele(m0, retires, appuis)
        info_k = dict(info, dos=info["dos"] & set(m.noeuds))
        # contact repris de l'itération précédente, deux passes au plus : l'ensemble actif du dos oscille sans
        # converger exactement, mais la contrainte utile n'en dépend pas (0,006 % de la 6e à la 40e passe, L + V)
        _contact(m, info_k, dossier, ITERATIONS_CONTACT, "eso_contact")
        appuis = set(m.appuis_x)
        res = lancer(m, dossier, "eso", details=True)
        _, sigma = pic_utile(res, info_k)
        masse = rho * sum(vol[e] for e in m.elements)
        effort = F * Re / sigma
        part = sum(vol[e] for e in retires) / v_ev
        etapes.append(EtapeESO(it, part, masse, sigma, effort, masse / (effort / 1000), sorted(retires)))
        afficher(f"ESO {it:2d} : {100 * part:3.0f} % retirés, {masse:.3f} kg, sigma utile {sigma:.0f} MPa, "
                 f"{etapes[-1].kg_par_kN:.4f} kg/kN")
        if part >= part_max - 1e-9:
            break
        champ = champ_elements(m, {e: faces_el[e] for e in m.elements}, res, epaisseur_face)
        vm = dict(zip((int(e) for e in champ.ids), champ.vm_peau))
        restants = [e for e in tous if e not in retires]
        filtre = {e: sum(w * vm[j] for j, w in voisins[e] if j in vm) / max(sum(w for j, w in voisins[e] if j in vm), 1e-12)
                  for e in restants}
        cible, cumul, nouveaux = pas * v_ev, 0.0, set()
        for e in sorted(restants, key=filtre.get):
            if cumul >= cible:
                break
            nouveaux.add(e)
            cumul += vol[e]
        while nouveaux and not _connexe({e: c for e, c in m0.elements.items() if e not in retires | nouveaux}):
            nouveaux.remove(max(nouveaux, key=filtre.get))  # garder d'un seul tenant : on rend les plus chargés
        if not nouveaux:
            break
        retires |= nouveaux

    out = ResultatESO(nom, taille, reference, etapes, evidable, exclues)
    # candidats : de vraies poches tirées de l'ESO à plusieurs niveaux, recalculées par la chaîne complète
    choisies = []
    for cible in niveaux:
        e = min((x for x in etapes if x.retires), key=lambda x: abs(x.part_retiree - cible), default=None)
        if e is not None and e not in choisies:
            choisies.append(e)
    for etape in choisies:
        out.candidats.append(candidat(concept, cdc, m0, faces_el, set(etape.retires), etape, taille, dossier, afficher))
    if not out.candidats:
        return out
    ok = [c for c in out.candidats if c.faisable and c.kg_par_kN < reference["kg_par_kN"]]
    out.retenu = min(ok, key=lambda c: c.kg_par_kN) if ok else None
    niveau = out.retenu or out.candidats[0]
    retenus = set(next(e for e in etapes if e.iteration == niveau.iteration).retires)
    for fid in evidable:
        f = concept.geometrie().face(fid)
        ids = [e for e, fs in faces_el.items() if fs == [fid]]
        codes = [0 if e in retenus else (1 if e in set(evidable[fid]) else 2) for e in ids]
        out.cartes[fid], out.axes[fid] = carte_face(f, np.array([centres[e] for e in ids]), np.array(codes),
                                                    {0: ".", 1: "o", 2: "#"})
    return out


def explication(reference: dict, c: Candidat) -> str:
    """Pourquoi un candidat gagne ou perd : kg/kN = masse x contrainte utile / (effort x Re), donc le rapport des
    kg/kN est exactement le produit du rapport des masses et de celui des contraintes utiles."""
    rm = c.masse_kg / reference["masse_kg"]
    rs = c.sigma_utile / reference["sigma_utile"]
    texte = (f"kg/kN x{rm * rs:.3f} = masse x{rm:.3f} x contrainte utile x{rs:.3f} "
             f"({reference['sigma_utile']:.0f} -> {c.sigma_utile:.0f} MPa)")
    p0, p1 = reference.get("point_critique"), c.point_critique
    if p0 and p1 and math.dist(p0, p1) > 10.0:
        texte += (f" ; point critique déplacé de {_lieu(reference.get('faces_critiques'), p0)} à "
                  f"{_lieu(c.faces_critiques, p1)}")
    elif p1:
        texte += f" ; point critique inchangé ({_lieu(c.faces_critiques, p1)})"
    return texte


def _lieu(faces: list[str] | None, p) -> str:
    return f"{'/'.join(faces or []) or 'hors face'} (x {p[0]:.0f}, y {p[1]:.0f}, z {p[2]:.0f})"


def _point(p) -> list[float] | None:
    return [round(float(x), 1) for x in p] if p is not None else None


def _participation(garde: dict, m0: ModeleCoque, faces_el: dict[int, list[str]], facteur: float) -> dict[str, float]:
    """Participation de chaque face au mode de flambement retenu (déplacement maximal / maximal global)."""
    rb = garde["flambement"]
    k = int(np.argmin([abs(f - facteur) if f > 0 else math.inf for f in rb.facteurs_flambement]))
    if k >= len(rb.modes):
        return {}
    u = {n: float(np.linalg.norm(v)) for n, v in rb.modes[k].items() if n not in m0.encastres}
    umax = max(u.values()) or 1.0
    par_face: dict[str, float] = {}
    for e, fs in faces_el.items():
        a = max(u.get(n, 0.0) for n in m0.elements[e]) / umax
        for f in fs:
            par_face[f] = max(par_face.get(f, 0.0), a)
    return par_face


def _aire(m: ModeleCoque, e: int) -> float:
    p = np.array([m.noeuds[n] for n in m.elements[e][:4]])
    return 0.5 * float(np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[1])))


def poches(concept: ConceptJSON, m0: ModeleCoque, faces_el: dict[int, list[str]], retires: set[int]) -> dict[str, list[Polygon]]:
    """Poches fabricables (repère plan de chaque face) tirées des éléments retirés par l'ESO."""
    geo = concept.geometrie()
    ctx = concept.contexte_evidements()
    par_face: dict[str, list[int]] = {}
    for e in retires:
        par_face.setdefault(faces_el[e][0], []).append(e)
    sortie: dict[str, list[Polygon]] = {}
    for fid, ids in par_face.items():
        f = geo.face(fid)
        rep = ev.Repere.de(f.points, f.normale())
        poly = Polygon(rep.vers_2d(f.points))
        t = f.t
        lig, r = ev.ligament_min(t), ev.rayon_defaut(t)
        zone = unary_union([Polygon(rep.vers_2d(np.array([m0.noeuds[n] for n in m0.elements[e][:4]]))).buffer(0.05)
                            for e in ids])
        zone = zone.buffer(r, join_style="round").buffer(-r, join_style="round")  # fermeture : trous et encoches comblés
        permis = poly.buffer(-lig).difference(ev.zones_protegees(f, ctx))
        for trou in f.trous:
            permis = permis.difference(Polygon(rep.vers_2d(np.asarray(trou))).buffer(lig))
        zone = zone.intersection(permis)
        zone = ev._relier_ilots(zone, poly, lig)
        zone = ev._ouverture(zone, r).buffer(-RETRAIT_EXPORT)
        gardes: list[Polygon] = []
        for p in sorted(ev._polygones(zone), key=lambda p: -p.area):
            if p.interiors:
                p = Polygon(p.exterior)  # un îlot resté au milieu d'une poche ne tiendrait à rien : retiré aussi
                if p.difference(permis.buffer(1e-6)).area > 1e-3:
                    continue
            if gardes:
                p = p.difference(unary_union(gardes).buffer(lig + 2 * RETRAIT_EXPORT))
                p = ev._ouverture(p, r)
                if p.is_empty:
                    continue
                p = ev._plus_grand(p)
            if p.area >= ev.AIRE_MIN and not p.interiors:
                gardes.append(p)
        if gardes:
            sortie[fid] = gardes
    return sortie


def concept_avec_poches(definition: dict, valeurs: dict, geo, poches_2d: dict[str, list[Polygon]],
                        suffixe: str = "allégé (ESO)") -> dict:
    """Concept figé aux valeurs données (chaque variable devient un choix à une seule valeur), avec les poches
    en « polygone » dans les faces : il se calcule, s'exporte et s'analyse comme les autres."""
    d = copy.deepcopy(definition)
    d["nom"] = f"{definition.get('nom', 'concept')} {suffixe}"
    d["description"] = (f"{definition.get('description', '')} Figé aux valeurs optimisées et allégé par optimisation "
                        "topologique (ESO) : poches tirées des zones peu sollicitées.").strip()
    d["variables"] = {k: {"type": "choix", "valeurs": [_nombre(valeurs[k])]} for k in definition["variables"]}
    for tole in d["toles"]:
        for fid, polys in poches_2d.items():
            nt, nf = fid.split(".", 1)
            if nt != tole["nom"]:
                continue
            f = geo.face(fid)
            rep = ev.Repere.de(f.points, f.normale())
            r = ev.rayon_defaut(f.t)
            for p in polys:
                pts = rep.vers_3d(np.asarray(p.exterior.coords)[:-1])
                tole.setdefault("evidements", []).append(
                    {"face": nf, "type": "polygone", "points": np.round(pts, 3).tolist(), "rayon": r})
    d.setdefault("justifications", {})["evidements_eso"] = (
        "Poches tirées d'une optimisation topologique ESO : on retire pas à pas la matière la moins sollicitée, puis "
        "on arrondit les zones retirées au rayon de fabrication en respectant les zones protégées.")
    return d


def candidat(concept: ConceptJSON, cdc: CahierDesCharges, m0, faces_el, retires: set[int], etape: EtapeESO,
             taille: float, dossier: Path, afficher=print) -> Candidat:
    geo = concept.geometrie()
    p2 = poches(concept, m0, faces_el, retires)
    d = concept_avec_poches(concept.definition, concept.valeurs, geo, p2)
    c = Candidat(etape.iteration, etape.part_retiree, d, {f: len(ps) for f, ps in p2.items()},
                 float(sum(p.area for ps in p2.values() for p in ps)))
    if not p2:
        c.erreurs = ["aucune poche fabricable à ce niveau"]
        return c
    allege = ConceptJSON(d, concept.valeurs, cdc)
    try:
        r = evaluer(allege, cdc, taille=taille, dossier=dossier / f"candidat_{etape.iteration}", flambement=True)
    except RuntimeError as ex:  # maillage invalide, calcul refusé : candidat écarté, la suite continue
        c.erreurs = [str(ex)[:300]]
        afficher(f"candidat ESO {100 * etape.part_retiree:.0f} % : écarté ({str(ex)[:120]})")
        return c
    c.erreurs = list(r.erreurs)
    if r.valide:
        c.masse_kg, c.effort_tenu_N, c.kg_par_kN = r.masse_kg, r.effort_tenu, r.masse_par_kN
        c.flambement, c.traction_vis_N = r.facteur_flambement, r.traction_vis_max
        c.sigma_utile, c.point_critique = r.sigma_max_utile, _point(r.point_sigma_utile)
        c.faces_critiques = allege.faces_proches(r.point_sigma_utile) if r.point_sigma_utile else []
        c.faisable = (r.effort_tenu >= cdc.chargements[0].force and math.isfinite(r.facteur_flambement)
                      and r.facteur_flambement >= FLAMBEMENT_MIN and r.traction_vis_max <= FT_RD_M10_88)
    afficher(f"candidat ESO {100 * etape.part_retiree:.0f} % : {sum(c.poches.values())} poches, "
             f"{c.kg_par_kN:.4f} kg/kN, flambement {c.flambement:.2f}, {'faisable' if c.faisable else 'non faisable'}")
    return c


def enregistrer(resultat: ResultatESO, dossier: Path) -> Path | None:
    """Écrit le texte du résultat et, s'il y en a un, le concept allégé retenu. Renvoie le chemin du concept."""
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "topologie.md").write_text(resultat.texte() + "\n", encoding="utf-8")
    for c in resultat.candidats:  # tous les candidats, pour pouvoir les analyser ou les reprendre
        (dossier / f"candidat_eso_{round(100 * c.part_retiree)}.json").write_text(
            json.dumps(c.definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if resultat.retenu is None:
        return None
    chemin = dossier / "concept_allege.json"
    chemin.write_text(json.dumps(resultat.retenu.definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return chemin


def _nombre(x):
    return int(x) if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e9) else float(x)
