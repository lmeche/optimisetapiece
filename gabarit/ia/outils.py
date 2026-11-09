"""Outils que l'IA Concepteur appelle pour contrôler ses concepts (indépendants du fournisseur d'IA).

`examiner_concept` construit le concept au point central de ses variables et sur des tirages aléatoires,
liste les erreurs (structure, géométrie, fabrication, soudures) et, si la géométrie est valide au centre,
lance un calcul statique rapide pour donner un ordre de grandeur (masse, effort tenu, zone la plus chargée).
"""
from __future__ import annotations

import collections
import math
import re
from pathlib import Path

import numpy as np

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts.format import ConceptJSON
from gabarit.etudes.concept_json import proposer_valeurs

TYPES_VARIABLES = {"epaisseur", "choix", "reel", "fraction", "retour", "evidement"}
TYPES_EVIDEMENTS = {"decalage", "cercle", "polygone"}
TAILLE_CALCUL_RAPIDE = 10.0
PART_TIRAGES_VALIDES_MIN = 0.5  # un tirage invalide ne coûte presque rien à l'optimiseur (rejeté avant maillage)


class _Tirage:
    """Imite un essai Optuna : point central (rng=None) ou tirage uniforme."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng

    def suggest_categorical(self, nom, choix):
        choix = list(choix)
        return choix[len(choix) // 2] if self.rng is None else choix[int(self.rng.integers(len(choix)))]

    def suggest_float(self, nom, bas, haut):
        return (bas + haut) / 2 if self.rng is None else float(self.rng.uniform(bas, haut))


def controler_structure(d) -> list[str]:
    """Erreurs de forme du fichier (avant toute construction géométrique)."""
    if not isinstance(d, dict):
        return ["le concept doit être un objet JSON"]
    e = []
    for cle, ty in (("nom", str), ("description", str), ("variables", dict), ("toles", list),
                    ("fixation", dict), ("appui", dict), ("justifications", dict)):
        if not isinstance(d.get(cle), ty):
            e.append(f"clé « {cle} » absente ou de mauvais type (attendu : {ty.__name__})")
    if e:
        return e
    noms_toles = [t.get("nom") for t in d["toles"] if isinstance(t, dict)]
    for nom, s in d["variables"].items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", nom):
            e.append(f"variable « {nom} » : nom invalide (lettres, chiffres, _)")
        if not isinstance(s, dict) or s.get("type") not in TYPES_VARIABLES:
            e.append(f"variable « {nom} » : type absent ou inconnu (types : {', '.join(sorted(TYPES_VARIABLES))})")
            continue
        ty = s["type"]
        if ty == "choix" and not (isinstance(s.get("valeurs"), list) and s["valeurs"]):
            e.append(f"variable « {nom} » : « valeurs » doit être une liste non vide")
        if ty == "reel" and not (isinstance(s.get("min"), (int, float)) and isinstance(s.get("max"), (int, float))
                                 and s["min"] < s["max"]):
            e.append(f"variable « {nom} » : « min » < « max » attendus")
        if ty in ("retour", "evidement"):
            if s.get("tole") not in noms_toles:
                e.append(f"variable « {nom} » : « tole » doit nommer une des tôles ({', '.join(map(str, noms_toles))})")
            if not (isinstance(s.get("max"), (int, float)) and s["max"] > 0):
                e.append(f"variable « {nom} » : « max » > 0 attendu")
    for t in d["toles"]:
        if not (isinstance(t, dict) and isinstance(t.get("nom"), str) and "epaisseur" in t
                and isinstance(t.get("faces"), dict) and t["faces"]):
            e.append("chaque tôle doit avoir « nom », « epaisseur » et des « faces »")
            continue
        e += _controler_evidements(t)
    if not isinstance(d.get("soudures", []), list):
        e.append("« soudures » doit être une liste de paires [\"tole.face\", \"tole.face\"]")
    if len(set(noms_toles)) != len(noms_toles):
        e.append("deux tôles portent le même nom")
    return e


def _controler_evidements(tole: dict) -> list[str]:
    e = []
    evs = tole.get("evidements", [])
    if not isinstance(evs, list):
        return [f"tôle {tole['nom']} : « evidements » doit être une liste"]
    for k, s in enumerate(evs, start=1):
        nom = f"tôle {tole['nom']}, évidement {k}"
        if not isinstance(s, dict) or s.get("type") not in TYPES_EVIDEMENTS or not isinstance(s.get("face"), str):
            e.append(f"{nom} : il faut « face » et « type » ({', '.join(sorted(TYPES_EVIDEMENTS))})")
            continue
        if s["type"] == "decalage" and "marge" not in s:
            e.append(f"{nom} : un « decalage » a besoin d'une « marge »")
        if s["type"] == "decalage" and not all(
                isinstance(n, list) and len(n) == 2 and all(isinstance(b, list) and len(b) == 2 for b in n)
                for n in s.get("nervures", [])):
            e.append(f"{nom} : chaque nervure s'écrit [[arête, position], [arête, position]]")
        if s["type"] == "cercle" and not (isinstance(s.get("centre"), list) and len(s["centre"]) == 3 and "diametre" in s):
            e.append(f"{nom} : un « cercle » a besoin de « centre » [x, y, z] et de « diametre »")
        if s["type"] == "polygone" and not (isinstance(s.get("points"), list) and len(s["points"]) >= 3):
            e.append(f"{nom} : un « polygone » a besoin d'au moins 3 « points »")
    return e


def examiner_concept(definition: dict, cdc: CahierDesCharges, n_tirages: int = 16, calcul: bool = True,
                     graine: int = 0) -> dict:
    """Rapport de contrôle d'un concept, sous une forme lisible par une IA comme par un humain."""
    erreurs = controler_structure(definition)
    if erreurs:
        return {"valide": False, "erreurs_structure": erreurs}
    try:
        centre = proposer_valeurs(definition, cdc, _Tirage())
    except Exception as ex:  # expression d'épaisseur fausse, tôle introuvable...
        return {"valide": False, "erreurs_structure": [f"variables inutilisables : {ex}"]}
    c = ConceptJSON(definition, centre, cdc)
    erreurs_centre = c.verifier(cdc)
    rapport = {"valide": False, "centre": {"valeurs": _arrondi(centre), "erreurs": erreurs_centre}}

    rng = np.random.default_rng(graine)
    compte = collections.Counter()
    valides = 0
    for _ in range(n_tirages):
        v = proposer_valeurs(definition, cdc, _Tirage(rng))
        err = ConceptJSON(definition, v, cdc).verifier(cdc)
        valides += not err
        compte.update({_generique(m) for m in err})
    part = valides / n_tirages
    rapport["tirages_aleatoires"] = {"nombre": n_tirages, "valides": valides,
                                     "erreurs_frequentes": [[m, k] for m, k in compte.most_common(8)]}
    if erreurs_centre:
        return rapport
    rapport["centre"]["masse_kg"] = round(c.masse(cdc.materiau.rho), 3)
    if calcul:
        rapport["centre"]["calcul_rapide"] = _calcul_rapide(c, cdc)
    rapport["valide"] = part >= PART_TIRAGES_VALIDES_MIN
    if not rapport["valide"]:
        rapport["a_corriger"] = (f"seulement {valides}/{n_tirages} tirages aléatoires sont valides "
                                 f"(il en faut {PART_TIRAGES_VALIDES_MIN:.0%}) : reparamétrer avec des fractions "
                                 "et des dérivées pour que toute combinaison des variables reste fabricable")
    return rapport


def _calcul_rapide(c: ConceptJSON, cdc: CahierDesCharges) -> dict:
    from gabarit.evaluation import evaluer  # import tardif : Gmsh et CalculiX ne servent qu'ici

    try:
        r = evaluer(c, cdc, taille=TAILLE_CALCUL_RAPIDE, flambement=False)
    except Exception as ex:
        return {"erreur": str(ex)[:300]}
    return {"maillage_mm": TAILLE_CALCUL_RAPIDE, "effort_tenu_N": round(r.effort_tenu),
            "objectif_requis_N": round(cdc.chargements[0].force), "sigma_utile_MPa": round(r.sigma_max_utile),
            "zone_sigma_utile": {"point_mm": r.point_sigma_utile,
                                 "faces": c.faces_proches(r.point_sigma_utile) if r.point_sigma_utile else []},
            "fleche_mm": round(r.fleche, 2), "traction_vis_max_N": round(r.traction_vis_max),
            "masse_par_kN": round(r.masse_par_kN, 3),
            "remarque": "calcul grossier au point central, sans flambement : ordre de grandeur seulement"}


def _generique(message: str) -> str:
    """Regroupe les messages qui ne diffèrent que par des nombres."""
    return re.sub(r"-?\d+(\.\d+)?", "#", message)


def _arrondi(v: dict) -> dict:
    return {k: (round(x, 2) if isinstance(x, float) and math.isfinite(x) else x) for k, x in v.items()}


def nom_fichier(definition: dict) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(definition.get("nom", "concept")).lower()
                  .translate(str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))).strip("_")
    return (base or "concept")[:60]


def enregistrer(definition: dict, dossier: Path) -> Path:
    import json

    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{nom_fichier(definition)}.json"
    k = 2
    while chemin.exists():
        chemin = dossier / f"{nom_fichier(definition)}_{k}.json"
        k += 1
    chemin.write_text(json.dumps(definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return chemin
