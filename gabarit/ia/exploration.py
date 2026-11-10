"""Exploration longue : un modèle open source enchaîne des sessions courtes d'IA Concepteur, chacune sur une
piste différente, et tient un catalogue de tout ce qui a été essayé.

Pensé pour un petit modèle qui tourne gratuitement sur le PC pendant des heures (une nuit) : il est moins bon
que Claude, mais il propose beaucoup plus de formes, et les contrôles de Gabarit écartent celles qui ne sont
pas fabricables. Chaque session repart d'une conversation neuve (le contexte d'un petit modèle reste court),
ne vise qu'un concept et s'arrête après quelques tours de correction. La piste vient d'abord d'une liste de
principes de structure ; quand elle est épuisée, le modèle invente des pistes nouvelles ou réessaie celles qui
ont échoué. Chaque concept enregistré est recalculé grossièrement (masse, effort tenu) pour le catalogue.

Sorties : concepts/exploration/<concept>.json, et dans resultats/exploration/<date>/ le catalogue
(catalogue.csv) et le journal de chaque essai (essai_001/journal.md...).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.ia.concepteur import RACINE, _relatif
from gabarit.ia.outils import examiner_concept
from gabarit.ia.ouvert import URL_OLLAMA, ErreurServeur, concevoir_ouvert
from gabarit.savoir.prediction import lire as lire_prediction

PISTES = [
    ("Équerre à joues dégressives : deux joues latérales dont la hauteur diminue du socle vers l'appui, reliées "
     "par le dessus."),
    ("Âme centrale : une tôle verticale dans le plan y = 0 sous l'appui, soudée au dos et à un dessus étroit "
     "(section en T)."),
    "Caisson fermé : dessus, dessous et deux flancs forment un tube rectangulaire en porte-à-faux depuis le dos.",
    "Cadre fermé triangulé : dos, dessus et diagonale faits d'une seule tôle pliée plusieurs fois.",
    ("Tirant suspendu : le dessus est retenu par une tôle tendue qui remonte vers les vis du haut, au lieu d'être "
     "soutenu par dessous."),
    "Une seule tôle pliée sans aucune soudure : dos, dessus et retours larges pour rigidifier.",
    "U inversé : le dessus et deux ailes tombantes sur toute la longueur, plus un dos plié.",
    "Deux diagonales en V vu de dessus, qui partent des bords du dos et se rejoignent sous l'appui.",
    ("Dessus raidi par des plis longitudinaux (forme en chapeau ou en oméga) pour porter l'effort jusqu'au dos "
     "sans diagonale."),
    "Gousset triangulaire central soudé sous le dessus et contre le dos.",
    "Jambes de force latérales inclinées qui partent du bas du dos et soutiennent les bords du dessus.",
    "Deux tôles en L identiques placées dos à dos et soudées sur leur partie commune.",
]
INVENTER = ("Invente un principe de structure nouveau, différent de toutes les pistes déjà essayées ci-dessous. "
            "Décris-le en une phrase au début de ta réponse et dans le champ « description » du concept.")
COLONNES = ["essai", "debut", "duree_min", "piste", "statut", "concept", "concept_nom", "masse_kg", "effort_tenu_kN",
            "kg_par_kN", "kg_par_kN_predit", "mode_predit", "principes", "valide_sur_tirages", "arret", "jetons_entree",
            "jetons_sortie"]
ECHECS_CONSECUTIFS_MAX = 3


def explorer(heures: float = 8.0, essais_max: int | None = None, url: str = URL_OLLAMA, modele: str = "qwen3:8b",
             cle: str = "", chemin_cdc: str | Path = RACINE / "cahiers" / "sae51_equerre.json",
             sortie: Path = RACINE / "concepts" / "exploration", dossier: Path | None = None,
             outils: bool = True, tours: int = 6, calcul: bool = True, delai: float = 1800.0,
             afficher=print) -> Path:
    """Lance des essais jusqu'à `heures` écoulées (ou `essais_max`) et renvoie le chemin du catalogue."""
    cdc = CahierDesCharges.charger(chemin_cdc)
    dossier = dossier or RACINE / "resultats" / "exploration" / time.strftime("%Y%m%d_%H%M%S")
    dossier.mkdir(parents=True, exist_ok=True)
    catalogue = dossier / "catalogue.csv"
    lignes: list[dict] = []
    debut = time.monotonic()
    echecs = 0
    i = 0
    while time.monotonic() - debut < heures * 3600 and (essais_max is None or i < essais_max):
        i += 1
        piste = choisir_piste(i, lignes)
        ecoule = (time.monotonic() - debut) / 60
        afficher(f"\n=== Essai {i} ({ecoule:.0f} min écoulées) : {piste[:120]} ===\n")
        ligne = {"essai": i, "debut": time.strftime("%H:%M:%S"), "piste": piste}
        t0 = time.monotonic()
        try:
            s = concevoir_ouvert(1, url=url, modele=modele, cle=cle, chemin_cdc=chemin_cdc,
                                 consigne=consigne(piste, lignes), sortie=sortie,
                                 dossier_journal=dossier / f"essai_{i:03d}", calcul=calcul, outils=outils,
                                 delai=delai, tours_max=tours, afficher=afficher)
        except (ErreurServeur, OSError) as ex:  # serveur arrêté, trop lent, clé refusée...
            echecs += 1
            ligne.update(statut="erreur", arret=str(ex)[:200])
            afficher(f"\n[erreur] {ex}\n")
            if isinstance(ex, ErreurServeur) and ex.code in (401, 403):
                echecs = ECHECS_CONSECUTIFS_MAX  # clé refusée : inutile d'insister
        else:
            echecs = 0
            outils = outils and not s.mode_texte  # inutile de redemander les outils à chaque essai
            ligne.update(arret=s.arret, jetons_entree=s.jetons["entree"], jetons_sortie=s.jetons["sortie"])
            if s.concepts:
                ligne.update(statut="enregistré", **_mesurer(s.concepts[-1], cdc, calcul))
            else:
                ligne["statut"] = "aucun concept"
        ligne["duree_min"] = round((time.monotonic() - t0) / 60, 1)
        lignes.append(ligne)
        _ecrire(catalogue, lignes)
        afficher(f"\n[catalogue] essai {i} : {ligne['statut']}"
                 + (f", {ligne['kg_par_kN']} kg/kN" if ligne.get("kg_par_kN") else "") + "\n")
        if echecs >= ECHECS_CONSECUTIFS_MAX:
            afficher(f"\n{echecs} échecs de suite : le serveur d'IA ne répond pas, exploration arrêtée.\n")
            break
    enregistres = [ligne for ligne in lignes if ligne["statut"] == "enregistré"]
    afficher(f"\n\nExploration finie : {len(lignes)} essais, {len(enregistres)} concepts enregistrés.\n"
             f"Catalogue : {catalogue}\n")
    return catalogue


def choisir_piste(i: int, lignes: list[dict]) -> str:
    """Les pistes de la liste d'abord ; ensuite une piste inventée une fois sur deux, sinon une piste de la
    liste qui n'a encore rien donné (le modèle peut réussir au second essai)."""
    if i <= len(PISTES):
        return PISTES[i - 1]
    reussies = {ligne["piste"] for ligne in lignes if ligne["statut"] == "enregistré"}
    a_reprendre = [p for p in PISTES if p not in reussies]
    if i % 2 or not a_reprendre:
        return INVENTER
    tentatives = {p: sum(ligne["piste"] == p for ligne in lignes) for p in a_reprendre}
    return min(a_reprendre, key=lambda p: (tentatives[p], PISTES.index(p)))


def consigne(piste: str, lignes: list[dict], n_max: int = 15) -> str:
    texte = f"Piste à explorer : {piste}"
    if lignes:
        deja = [f"- {_nom_piste(ligne)} : {ligne['statut']}"
                + (f", {ligne['kg_par_kN']} kg/kN au calcul rapide" if ligne.get("kg_par_kN") else "")
                for ligne in lignes[-n_max:]]
        texte += "\n\nDéjà essayé (ne refais pas la même chose) :\n" + "\n".join(deja)
    return texte


def _nom_piste(ligne: dict) -> str:
    if ligne["piste"] == INVENTER:
        return f"piste inventée « {ligne.get('concept_nom') or 'sans résultat'} »"
    return ligne["piste"].split(" :")[0]


def _mesurer(chemin: Path, cdc: CahierDesCharges, calcul: bool) -> dict:
    """Masse et effort tenu au point central (calcul grossier), pour classer les concepts du catalogue."""
    definition = json.loads(chemin.read_text(encoding="utf-8"))
    m = {"concept": _relatif(chemin), "concept_nom": definition.get("nom", chemin.stem)}
    pred, _ = lire_prediction(definition)
    if pred:  # prédiction physique de l'IA (au calcul complet, dimensions optimisées) : à vérifier plus tard
        m["kg_par_kN_predit"] = pred.get("kg_par_kN")
        m["mode_predit"] = pred.get("mode_limite")
        m["principes"] = " ".join(pred.get("principes", []))
    try:
        r = examiner_concept(definition, cdc, calcul=calcul)
    except Exception as ex:  # le concept est déjà enregistré : une mesure ratée ne doit pas arrêter la nuit
        return {**m, "arret": f"mesure impossible : {str(ex)[:150]}"}
    centre = r.get("centre", {})
    t = r.get("tirages_aleatoires", {})
    m["masse_kg"] = centre.get("masse_kg")
    m["valide_sur_tirages"] = f"{t.get('valides')}/{t.get('nombre')}" if t else None
    rapide = centre.get("calcul_rapide") or {}
    if rapide.get("effort_tenu_N"):
        m["effort_tenu_kN"] = round(rapide["effort_tenu_N"] / 1000, 2)
        m["kg_par_kN"] = rapide.get("masse_par_kN")
    return m


def _ecrire(chemin: Path, lignes: list[dict]) -> None:
    with chemin.open("w", encoding="utf-8-sig", newline="") as f:  # -sig : accents lus par Excel
        w = csv.DictWriter(f, fieldnames=COLONNES, extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)
