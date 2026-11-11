"""Outil « predimensionner » de l'IA Concepteur : une calculatrice de physique des tôles.

L'IA décrit son idée en quelques nombres (hauteur du couple, bielles et leur section, parois comprimées,
soudures) ; l'outil rend les efforts, les contraintes, les facteurs de flambement et de voilement, la borne du
kg/kN et ce qui coince, avec les numéros des principes en jeu. Un petit modèle calcule mal de tête : l'outil
fait l'arithmétique, le modèle garde le raisonnement. Aucun calcul par éléments finis ici (réponse immédiate).
"""
from __future__ import annotations

import math

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.savoir import calculs as ca
from gabarit.savoir.principes import hauteur_disponible

_SECTION = {
    "type": "object",
    "description": "{\"type\": \"plaque\", \"b\": largeur, \"t\": épaisseur} ou {\"type\": \"U\", \"ame\": .., \"aile\": .., \"t\": ..} (mm)",
}
SCHEMA = {
    "type": "object",
    "properties": {
        "treillis": {"type": "object", "description": (
            "Couple tirant / bielle : {\"h\": écart vertical entre les attaches du tirant et de la bielle sur le socle "
            "(mm), \"L\": porte-à-faux (mm, défaut : celui du cahier des charges), \"disposition\": \"tirant_horizontal\" "
            "ou \"symetrique\"}")},
        "bielles": {"type": "array", "description": (
            "Barres comprimées : {\"nom\", \"longueur\" (mm), \"section\", \"effort_N\" (défaut : la bielle du treillis), "
            "\"K\" (longueur de flambement / longueur, défaut 0,6 pour une barre soudée aux deux bouts)}"),
            "items": {"type": "object", "properties": {"section": _SECTION}}},
        "parois": {"type": "array", "description": (
            "Parois comprimées à contrôler : {\"nom\", \"b\" largeur libre (mm), \"t\" (mm), \"bords\": 2 si ses deux "
            "bords longs sont tenus (pli, soudure), 1 si un bord est libre, \"contrainte_MPa\" (facultatif)}"),
            "items": {"type": "object"}},
        "soudures": {"type": "array", "description": (
            "Soudures d'angle : {\"nom\", \"effort_N\", \"longueur\" (mm, par cordon), \"cordons\" (défaut 2)}"),
            "items": {"type": "object"}},
    },
}
DESCRIPTION = (
    "Calculatrice de prédimensionnement, sans éléments finis, réponse immédiate. Donne les efforts du couple "
    "tirant / bielle pour une hauteur h, la borne du kg/kN, et pour chaque bielle sa contrainte sous la charge, "
    "ses facteurs de flambement d'ensemble et de voilement (exigence : 3), sa masse et l'écrouissage de ses plis ; "
    "pour chaque paroi comprimée son b/t face aux limites ; pour chaque soudure la gorge nécessaire. Unités : N, mm, "
    "MPa. À appeler pour chiffrer une idée avant d'écrire le concept et sa prédiction.")


def _nombre(d: dict, cle: str, defaut=None, positif: bool = True) -> float:
    v = d.get(cle, defaut)
    if v is None:
        raise ValueError(f"« {cle} » manquant")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"« {cle} » doit être un nombre") from None
    if not math.isfinite(x) or (positif and x <= 0):
        raise ValueError(f"« {cle} » doit être un nombre positif")
    return x


def _r(x: float, n: int = 2):
    return round(x, n) if math.isfinite(x) else None


def predimensionner(demande: dict, cdc: CahierDesCharges) -> dict:
    if not isinstance(demande, dict):
        return {"erreurs": ["l'entrée doit être un objet JSON"]}
    m, ch = cdc.materiau, cdc.chargements[0]
    F, rho, Re, E, nu = ch.force, m.rho, m.Re, m.E, m.nu
    fu = ca.rm_min(m.nom)
    rapport: dict = {"charge_N": F, "materiau": f"{m.nom}, Re {Re:g} MPa, Rm mini {fu:g} MPa", "erreurs": [], "alertes": []}
    bielle_treillis = None

    if "treillis" in demande:
        d = demande["treillis"] if isinstance(demande["treillis"], dict) else {}
        try:
            L = _nombre(d, "L", ch.distance_encastrement)
            h = _nombre(d, "h")
            disp = str(d.get("disposition") or "tirant_horizontal")
            t = ca.treillis(F, L, h, disp)
            hd = hauteur_disponible(cdc)
            bielle_treillis = t
            rapport["treillis"] = {
                "h_mm": h, "L_mm": L, "disposition": disp, "tirant_N": round(t.tirant_N), "bielle_N": round(t.bielle_N),
                "longueur_bielle_mm": _r(t.l_bielle, 1), "L_eq_mm": round(t.l_eq),
                "borne_kg_par_kN": _r(t.borne_kg_par_kN(rho, Re), 4),
                "h_optimal_mm": round(ca.hauteur_optimale(L, disp)), "h_disponible_mm": hd["h_mm"],
                "lecture": ("borne d'un treillis où toute la matière travaille à Re (eta = 1). Pièce réelle : "
                            "kg/kN = borne x (L_eq réelle / L_eq du treillis) / eta ; le L + V mesuré a L_eq 1 852 mm "
                            "et eta 0,22 (principes P1, P6)")}
            if h > hd["h_mm"] + 30:
                rapport["alertes"].append(f"h = {h:g} mm dépasse nettement la place entre la semelle et les vis "
                                          f"({hd['h_mm']:g} mm) : la traction doit redescendre jusqu'aux vis")
        except ValueError as ex:
            rapport["erreurs"].append(f"treillis : {ex}")

    rapport["bielles"] = []
    for i, d in enumerate(demande.get("bielles") or []):
        nom = str(d.get("nom") or f"bielle {i + 1}") if isinstance(d, dict) else f"bielle {i + 1}"
        try:
            if not isinstance(d, dict):
                raise TypeError("chaque bielle est un objet")
            s = ca.section_depuis(d.get("section") or {})
            longueur = _nombre(d, "longueur")
            effort = _nombre(d, "effort_N", bielle_treillis.bielle_N if bielle_treillis else None)
            K = _nombre(d, "K", ca.K_SOUDE)
            b = ca.barre_comprimee(nom, s, longueur, effort, E, nu, Re, rho, K)
            r = {"nom": nom, "section": s.nom, "aire_mm2": _r(s.A, 1), "inertie_faible_mm4": round(s.I_faible),
                 "effort_N": round(b.effort_N), "contrainte_MPa": _r(b.contrainte, 1),
                 "facteur_resistance": _r(b.facteur_resistance), "facteur_flambement_ensemble": _r(b.facteur_euler),
                 "voilement": [{"paroi": p, "sigma_cr_MPa": round(sc), "facteur": _r(f)} for p, sc, f in b.voilement],
                 "facteur_flambement": _r(b.facteur_flambement), "mode": b.mode(), "masse_kg": _r(b.masse_kg, 3),
                 "flambement_suffisant": b.facteur_flambement >= ca.COEF_FLAMBEMENT_MIN}
            if s.n_plis:
                r["re_ecroui_MPa"] = _r(ca.ecrouissage(Re, fu, s.parois[0][2], s.n_plis, s.A), 1)
            if b.facteur_flambement < ca.COEF_FLAMBEMENT_MIN:
                r["a_faire"] = ("plier (retours, U) ou raccourcir : principe P3" if s.plaque
                                else "élargir les ailes, épaissir ou raidir la paroi qui voile : principe P3")
            rapport["bielles"].append(r)
        except (ValueError, KeyError, TypeError) as ex:
            rapport["erreurs"].append(f"{nom} : {ex}")

    rapport["parois"] = []
    for i, d in enumerate(demande.get("parois") or []):
        nom = str(d.get("nom") or f"paroi {i + 1}") if isinstance(d, dict) else f"paroi {i + 1}"
        try:
            if not isinstance(d, dict):
                raise TypeError("chaque paroi est un objet")
            b, t = _nombre(d, "b"), _nombre(d, "t")
            bords = 1 if int(_nombre(d, "bords", 2)) < 2 else 2
            lim = ca.elancement_max(Re, bords, E, nu)
            sc = ca.voilement(t, b, bords, E, nu)
            r = {"nom": nom, "b_sur_t": _r(b / t, 1), "bords_tenus": bords,
                 "b_sur_t_max_eurocode": round(lim["eurocode_classe_3"]), "b_sur_t_max_elastique": round(lim["elastique"]),
                 "sigma_cr_MPa": round(sc), "trapue": b / t <= lim["eurocode_classe_3"]}
            if "contrainte_MPa" in d:
                r["facteur_voilement"] = _r(sc / _nombre(d, "contrainte_MPa"))
            if not r["trapue"]:
                r["a_faire"] = ("ajouter un retour sur le bord libre (P3)" if bords == 1
                                else "raidir (pli intermédiaire, raidisseur) ou épaissir (P3)")
            rapport["parois"].append(r)
        except (ValueError, TypeError) as ex:
            rapport["erreurs"].append(f"{nom} : {ex}")

    rapport["soudures"] = []
    beta = ca.BETA_W.get(m.nom.upper().split()[0], 0.8)
    for i, d in enumerate(demande.get("soudures") or []):
        nom = str(d.get("nom") or f"soudure {i + 1}") if isinstance(d, dict) else f"soudure {i + 1}"
        try:
            if not isinstance(d, dict):
                raise TypeError("chaque soudure est un objet")
            a = ca.gorge_min(_nombre(d, "effort_N"), _nombre(d, "longueur"), fu, beta, int(_nombre(d, "cordons", 2)))
            rapport["soudures"].append({"nom": nom, "f_vw_d_MPa": round(ca.fvw_d(fu, beta), 1), "gorge_min_mm": _r(a),
                                        "gorge_retenue_mm": max(3.0, math.ceil(a))})
        except (ValueError, TypeError) as ex:
            rapport["erreurs"].append(f"{nom} : {ex}")

    for cle in ("bielles", "parois", "soudures", "erreurs", "alertes"):
        if not rapport[cle]:
            del rapport[cle]
    if len(rapport) <= 2:
        rapport["aide"] = "rien à calculer : donne au moins « treillis », « bielles », « parois » ou « soudures »"
    return rapport
