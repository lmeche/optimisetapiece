"""Prédiction physique d'un concept : ce que l'IA annonce AVANT le calcul, que Gabarit compare ensuite au calcul.

Champ facultatif « prediction » d'un concept :

    "prediction": {"kg_par_kN": 0.24, "mode_limite": "plastification", "face_critique": "L.dessus", "h_mm": 150,
                   "effort_tenu_kN": 16, "principes": ["P1", "P3"], "raisonnement": "en une ou deux phrases"}

- kg_par_kN : la valeur attendue au calcul complet (obligatoire) ;
- mode_limite : la plus petite des trois marges mesurées par Gabarit, « plastification » (effort tenu / 1 t),
  « flambement » (facteur / 3) ou « vis » (résistance / traction max par vis) (obligatoire) ;
- face_critique : la face (« tôle.face ») où sera le point critique (facultatif, jugé à part : « lieu ») ;
- principes : les principes de gabarit/savoir/principes.py sur lesquels repose la prédiction ;
- h_mm, effort_tenu_kN, raisonnement : facultatifs, gardés pour la lecture.

Verdict (au calcul de la fiche) : la valeur est juste si l'écart relatif sur le kg/kN est au plus de
TOLERANCE_PCT, le mode est juste s'il est celui de la fiche ; la prédiction est « juste » si les deux le sont,
« partielle » si un seul l'est, « fausse » sinon. La tolérance laisse place au bruit du maillage (quelques %)
et à la précision d'un calcul à la main ; l'écart signé est gardé pour mesurer le biais de chaque principe.
"""
from __future__ import annotations

import math

from gabarit.savoir.principes import IDS

TOLERANCE_PCT = 15.0
MODES = ("plastification", "flambement", "vis")


def mode_de_limite(limite: str) -> str:
    """Mode d'une fiche, tiré de sa limite (gabarit/analyse/compte_rendu.py : _limite)."""
    t = (limite or "").lower()
    if "flamb" in t and "non calculé" not in t:
        return "flambement"
    if "vis" in t:
        return "vis"
    if "plastif" in t:
        return "plastification"
    return ""


def lire(definition: dict) -> tuple[dict | None, list[str]]:
    """Prédiction normalisée d'un concept, et ce qui y manque. (None, []) si le concept n'en a pas."""
    x = definition.get("prediction") if isinstance(definition, dict) else None
    if x is None:
        return None, []
    if not isinstance(x, dict):
        return None, ["« prediction » doit être un objet"]
    erreurs = []
    p: dict = {}
    try:
        p["kg_par_kN"] = float(x.get("kg_par_kN"))
        if not (math.isfinite(p["kg_par_kN"]) and p["kg_par_kN"] > 0):
            raise ValueError
    except (TypeError, ValueError):
        erreurs.append("« kg_par_kN » : nombre positif attendu")
        p.pop("kg_par_kN", None)
    mode = str(x.get("mode_limite") or "").strip().lower()
    mode = next((m for m in MODES if mode.startswith(m[:5])), "")
    if mode:
        p["mode_limite"] = mode
    else:
        erreurs.append(f"« mode_limite » : un de {', '.join(MODES)}")
    for cle in ("h_mm", "effort_tenu_kN"):
        try:
            v = float(x[cle]) if x.get(cle) is not None else math.nan
            if math.isfinite(v):
                p[cle] = v
        except (TypeError, ValueError):
            erreurs.append(f"« {cle} » : nombre attendu")
    principes = x.get("principes") or []
    principes = [str(i).strip().upper() for i in (principes if isinstance(principes, list) else [principes])]
    inconnus = [i for i in principes if i not in IDS]
    if inconnus:
        erreurs.append(f"principes inconnus : {', '.join(inconnus)} (P1 à P9)")
    face = str(x.get("face_critique") or "").strip()
    if face:
        p["face_critique"] = face
    p["principes"] = sorted({i for i in principes if i in IDS}, key=lambda i: int(i[1:]))
    p["raisonnement"] = str(x.get("raisonnement") or "").strip()[:600]
    return p, erreurs


def juger(kg_predit: float, mode_predit: str, kg_mesure: float, mode_mesure: str,
          tolerance_pct: float = TOLERANCE_PCT, face_predite: str = "", face_mesuree: str = "") -> tuple[str, dict]:
    """Verdict et mesures d'une prédiction face à une fiche. Le lieu (face du point critique), s'il est prédit,
    est jugé à part et n'entre pas dans le verdict."""
    ecart = 100 * (kg_predit - kg_mesure) / kg_mesure if kg_mesure and math.isfinite(kg_mesure) else math.nan
    valeur_juste = math.isfinite(ecart) and abs(ecart) <= tolerance_pct
    mode_juste = bool(mode_mesure) and mode_predit == mode_mesure
    verdict = "juste" if valeur_juste and mode_juste else ("partielle" if valeur_juste or mode_juste else "fausse")
    mesure = {"kg_par_kN": kg_mesure, "mode_limite": mode_mesure, "ecart_pct": ecart,
              "valeur_juste": valeur_juste, "mode_juste": mode_juste}
    if face_predite:
        mesure.update(face_critique=face_mesuree, lieu_juste=bool(face_mesuree) and face_predite == face_mesuree)
    return verdict, mesure
