"""Tests hors ligne de l'apprentissage (sans CalculiX, sans clé API, sans réseau) : statistiques, diagnostic de
la zone d'appui du compte rendu, carnet
(fiches, verdicts, statuts, enregistrement), IA Analyste (contrôle de ses réponses, faux serveurs Claude et
modèle ouvert), reprise des cotes d'un parent, boucle d'apprentissage avec des doublures des étapes coûteuses.

Lancer :  python tests/tester_apprentissage.py
"""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from scipy import stats as sps

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
import anthropic
import optuna

from gabarit.analyse.compte_rendu import BilanFace, CompteRendu
from gabarit.apprentissage import analyste as an
from gabarit.apprentissage import boucle as bo
from gabarit.apprentissage import statistiques as st
from gabarit.apprentissage.carnet import Carnet, verdict
from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.etudes.concept_json import (
    parametres_depuis_valeurs,
    proposer_valeurs,
)
from gabarit.ia.concepteur import Session

CDC = CahierDesCharges.charger(RACINE / "cahiers" / "sae51_equerre.json")
LV = RACINE / "concepts" / "sae51_L_V.json"
VALEURS = json.loads((RACINE / "tests" / "donnees" / "sae51_L_V_optimum.json").read_text(encoding="utf-8"))["valeurs"]
RHO, RE, F = 7.85e-6, 235.0, 9810.0  # kg/mm³, MPa, N
resultats: list[tuple[str, bool, str]] = []


def verifier(nom: str, ok: bool, detail: str = "") -> None:
    resultats.append((nom, bool(ok), detail))


def cr_factice(nom: str, l_eq: float, eta: float, masse: float = 4.0, flambement: float = 3.3,
               marges: dict | None = None, valide: bool = True) -> CompteRendu:
    """Compte rendu cohérent : kg/kN = 1000 rho/Re x L_eq / eta exactement, effort tenu = masse / kg/kN."""
    kg = 1000 * RHO / RE * l_eq / eta
    effort = masse / kg * 1000
    sigma = RE * F / effort
    if not valide:
        return CompteRendu(nom, {"t_L": 5.0}, 6.0, False, ["maillage impossible"])
    m = {"masse_kg": masse, "effort_tenu_N": effort, "kg_par_kN": kg, "sigma_utile": sigma, "fleche": 0.4,
         "flambement": flambement, "traction_vis_N": 11000.0, "effort_N": F, "rho": RHO, "Re": RE}
    marges = marges or {"plastification": effort / F, "flambement": flambement / 3.0, "vis": 33408.0 / 11000.0}
    faces = [BilanFace("L.dessus", 5.0, 1e5, 0.4 * masse, 0.4, 0.5, 0.5 * l_eq, eta, 0.3, sigma, 0.6),
             BilanFace("V.diag", 5.0, 5e4, 0.6 * masse, 0.6, 0.5, 0.5 * l_eq, eta, 0.1, 0.8 * sigma, 0.1)]
    return CompteRendu(nom, {"t_L": 5.0}, 6.0, True, [], mesures=m, marges=marges, limite="la plastification",
                       l_eq_mm=l_eq, eta=eta, part_flexion=0.25, point_critique={"face": "L.dessus", "texte": "pli"},
                       faces=faces, pistes=["piste automatique"])


def test_statistiques() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 4, 9).astype(float)  # ex aequo
    verifier("stats : rangs = scipy.stats.rankdata", np.allclose(st.rangs(x), sps.rankdata(x)))
    y = x + rng.normal(0, 1, 9)
    verifier("stats : Spearman = scipy.stats.spearmanr", math.isclose(st.spearman(x, y), sps.spearmanr(x, y)[0]))
    x6, y6 = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]
    rho, p = st.spearman_p(x6, y6)
    brut = [abs(st.spearman(x6, [y6[i] for i in perm])) >= abs(rho) - 1e-12 for perm in itertools.permutations(range(6))]
    verifier("stats : valeur p exacte = dénombrement des 720 permutations", math.isclose(p, np.mean(brut)),
             f"rho {rho:.3f}, p {p:.4f}")
    _, p_tir = st.spearman_p(list(range(12)), [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10])
    verifier("stats : valeur p par tirages sur 12 concepts, forte corrélation", p_tir < 0.001, f"p {p_tir:.5f}")
    verifier("stats : Holm (exemple fait à la main)",
             np.allclose(st.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02]))
    l_eq, eta = np.array([1800, 2100, 1500, 2600.0]), np.array([0.23, 0.2, 0.25, 0.3])
    d = st.decomposition_variance(1000 * RHO / RE * l_eq / eta, l_eq, eta)
    verifier("stats : parts de variance (L_eq, eta, covariance) de somme 1, identité exacte",
             math.isclose(d["l_eq"] + d["eta"] + d["covariance"], 1.0) and d["ecart_identite"] < 1e-12,
             f"{d['l_eq']:.2f} + {d['eta']:.2f} + {d['covariance']:.2f}")


class ResultatsFactices:
    """Le peu qu'utilise le diagnostic de la zone d'appui : Von Mises et coordonnées des noeuds."""

    def __init__(self, vm: dict, coords: dict):
        self._vm, self.coords = vm, coords

    def von_mises(self) -> dict:
        return dict(self._vm)


def test_zone_appui() -> None:
    from gabarit.analyse import compte_rendu as crm

    info = {"x_charge": 320.0, "z_charge": 192.0, "t_charge": 5.0, "x_dos": 20.0, "t_dos": 5.0, "trous": [],
            "singuliers": [(200.0, 0.0, 192.0)]}
    coords = {1: (351.0, 0.0, 190.0), 2: (300.0, 0.0, 170.0), 3: (100.0, 40.0, 192.0), 4: (200.0, 0.0, 192.0),
              5: (330.0, 0.0, 192.0)}
    vm = {1: 146.0, 2: 140.0, 3: 96.0, 4: 300.0, 5: 500.0}  # 4 : extrémité de jonction, 5 : sous l'appui (exclus)
    z = crm._zone_appui(ResultatsFactices(vm, coords), info, np.array(coords[1]), 146.0)
    verifier("compte rendu : point critique dans la zone d'appui, reste de la pièce lu hors singularités",
             z["critique_dans_zone"] and math.isclose(z["sigma_hors_zone"], 96.0)
             and math.isclose(z["rapport"], 146 / 96) and z["xyz"] == [100.0, 40.0, 192.0], json.dumps(z))
    z["face"] = "L.dos"
    texte = crm._texte_zone_appui(z)
    verifier("compte rendu : zone d'appui en clair", "à 31 mm du centre de la charge" in texte and "x1.52" in texte
             and "96 MPa au plus (L.dos" in texte, texte)
    cr = cr_factice("A", 1800.0, 0.22)
    cr.point_critique["appui"] = z
    verifier("compte rendu : piste « étaler l'effort » quand la zone d'appui décide",
             any("zone d'introduction de l'effort fixe" in t for t in crm.pistes(cr)))
    loin = crm._zone_appui(ResultatsFactices({3: 96.0, 2: 50.0}, coords), info, np.array(coords[3]), 96.0)
    loin["face"] = "L.dos"
    cr.point_critique["appui"] = loin
    verifier("compte rendu : point critique loin de l'appui, ni constat ni piste",
             not loin["critique_dans_zone"] and "hors de la zone d'appui" in crm._texte_zone_appui(loin)
             and not any("zone d'introduction" in t for t in crm.pistes(cr)))


def carnet_de_base(dossier: Path) -> Carnet:
    c = Carnet(dossier / "carnet.json")
    c.ajouter_fiche(cr_factice("L + V", 1828.0, 0.233, 4.376), fichier=LV, convergence={"taille": 8.0, "kg_par_kN": 0.25})
    c.ajouter_fiche(cr_factice("Variante", 2100.0, 0.22, 4.1), fichier=LV, convergence={"taille": 8.0, "kg_par_kN": 0.30})
    return c


def test_carnet(dossier: Path) -> None:
    c = carnet_de_base(dossier)
    f1, _ = c.fiches
    verifier("carnet : fiche faisable, marges et décomposition copiées",
             f1.faisable and math.isclose(f1.marges["flambement"], 1.1) and f1.decomposition["eta"] == 0.233, f1.resume())
    verifier("carnet : meilleure fiche = plus petit kg/kN", c.meilleures(1)[0].id == "F001")
    verifier("verdict : baisse de 5 % confirmée, hausse de 5 % contredite, 2 % non concluante, NaN non concluante",
             [verdict(1.0, x, "baisse", 3.0)[0] for x in (0.95, 1.05, 0.98, math.nan)]
             == ["confirmee", "contredite", "non_concluante", "non_concluante"])

    lecon = c.ajouter_lecon("Un chemin d'effort court baisse le kg/kN", "moins de V x sigma", "toujours")
    p = c.preuve_comparaison(lecon, "F001", "F002", "kg_par_kN", "<")
    verifier("preuve : valeurs lues dans les fiches, comparaison conforme, leçon observée",
             p.verdict == "conforme" and p.valeurs[0] == c.fiche("F001").mesures["kg_par_kN"] and lecon.statut == "observee")
    l2 = c.ajouter_lecon("Leçon que les mesures contredisent")
    c.preuve_comparaison(l2, "F001", "F002", "kg_par_kN", ">")
    verifier("preuve : comparaison contraire, la leçon reste une hypothèse", l2.statut == "hypothese")

    def experience(cr, sens="baisse", lecon_id=None, grandeur="kg_par_kN"):
        e = c.ajouter_experience("F001", "modification", grandeur, sens, lecon_id)
        f = c.ajouter_fiche(cr, fichier=LV, parent="F001", experience=e.id) if cr is not None else None
        return c.conclure_experience(e.id, f.id if f else None)

    e = experience(cr_factice("mieux", 1700.0, 0.233, 4.2), lecon_id=lecon.id)
    verifier("expérience : enfant faisable 7 % meilleur, confirmée, la leçon passe à « confirmée »",
             e.verdict == "confirmee" and lecon.statut == "confirmee", e.texte())
    e = experience(cr_factice("moins bien", 1950.0, 0.233, 4.4), lecon_id=lecon.id)
    verifier("expérience : enfant moins bon, contredite ; leçon avec les deux résultats « nuancée »",
             e.verdict == "contredite" and lecon.statut == "nuancee")
    e = experience(cr_factice("flambe", 1600.0, 0.233, 4.0, flambement=2.4))
    verifier("expérience : enfant nettement non faisable (flambement x0,8) : contredite malgré un kg/kN plus bas",
             e.verdict == "contredite" and e.mesure["ecart_pct"] < -3 and "faisabilité" in e.mesure["note"], e.texte())
    e = experience(cr_factice("limite", 1700.0, 0.233, 4.0, flambement=2.98))
    verifier("expérience : enfant à moins du seuil de sa limite (x0,993) : verdict sur le kg/kN mesuré, noté",
             e.verdict == "confirmee" and "seuil" in e.mesure["note"], e.texte())
    e = experience(cr_factice("flambe", 1950.0, 0.233, 4.0, flambement=2.4), sens="hausse")
    verifier("expérience : kg/kN prédit en hausse, enfant non faisable : confirmée", e.verdict == "confirmee")
    e = experience(cr_factice("inconnu", 1600.0, 0.233, 4.0, flambement=math.nan))  # calcul de flambement raté
    verifier("expérience : marge de flambement inconnue : non concluante", e.verdict == "non_concluante", e.texte())
    e = experience(cr_factice("cassé", 1.0, 1.0, valide=False))
    verifier("expérience : enfant non calculable : abandonnée", e.statut == "abandonnee" and e.verdict == "echec")
    e = experience(None)
    verifier("expérience : pas d'enfant : abandonnée", e.statut == "abandonnee")
    e = experience(cr_factice("flambement", 1828.0, 0.233, 4.376, flambement=3.6), sens="hausse", grandeur="flambement")
    verifier("expérience : autre grandeur (flambement en hausse de 9 %) : confirmée", e.verdict == "confirmee")

    def experience_deux_mailles(kg8_enfant):
        c.fiche("F001").convergence = {"taille": 8.0, "kg_par_kN": 0.26}
        e = c.ajouter_experience("F001", "modification", "kg_par_kN", "baisse")
        f = c.ajouter_fiche(cr_factice("mieux à 6 mm", 1700.0, 0.233, 4.2), fichier=LV, parent="F001",
                            experience=e.id, convergence={"taille": 8.0, "kg_par_kN": kg8_enfant})
        return c.conclure_experience(e.id, f.id)

    e = experience_deux_mailles(0.262)
    verifier("expérience : 7 % meilleure à 6 mm mais +0,8 % à 8 mm : non concluante (l'écart dépend de la maille)",
             e.verdict == "non_concluante" and "dépend de la maille" in e.mesure.get("note", ""), e.texte())
    e = experience_deux_mailles(0.245)
    verifier("expérience : meilleure aux deux mailles : confirmée, écart à 8 mm noté",
             e.verdict == "confirmee" and math.isclose(e.mesure["ecart_controle_pct"], 100 * (0.245 / 0.26 - 1)),
             e.texte())
    c.fiche("F001").convergence = {}
    try:
        c.ajouter_experience("F999", "x")
        refus = False
    except ValueError:
        refus = True
    verifier("expérience : parent inconnu refusé", refus)

    c.cycles = 7
    c.enregistrer()
    relu = Carnet.charger(c.chemin)
    verifier("carnet : relu à l'identique (fiches, leçons, expériences, cycles)",
             [f.__dict__ for f in relu.fiches] == [f.__dict__ for f in c.fiches]
             and [x.__dict__.keys() for x in relu.lecons] == [x.__dict__.keys() for x in c.lecons]
             and [p.__dict__ for x in relu.lecons for p in x.preuves] == [p.__dict__ for x in c.lecons for p in x.preuves]
             and [e.__dict__ for e in relu.experiences] == [e.__dict__ for e in c.experiences] and relu.cycles == 7)
    texte = c.chemin.with_suffix(".md").read_text(encoding="utf-8")
    verifier("carnet : version lisible écrite à côté", "| F001 |" in texte and "nuancee" in texte)
    json.loads(c.chemin.read_text(encoding="utf-8"))  # JSON strict (pas de NaN)
    verifier("carnet : JSON strict, sans NaN", "NaN" not in c.chemin.read_text(encoding="utf-8"))

    prompt = c.pour_prompt(parent="F001")
    verifier("prompt : leçons par statut, meilleurs concepts, modifications déjà essayées sur le parent",
             "Nuancées" in prompt and "Meilleurs concepts" in prompt and "déjà essayées" in prompt
             and "F001" in prompt, f"{len(prompt)} caractères")
    verifier("prompt compact plus court", len(c.pour_prompt(compact=True, parent="F001")) < len(prompt))


def test_statistiques_carnet(dossier: Path) -> None:
    c = Carnet(dossier / "stats.json")
    rng = np.random.default_rng(3)
    for k in range(7):
        l_eq, eta = 1500 + 150 * k, 0.2 + 0.01 * rng.random()
        f = c.ajouter_fiche(cr_factice(f"c{k}", l_eq, eta), fichier=LV)
        f.caracteristiques = {"n_toles": 2 + k % 2, "n_faces": 5 + k, "n_plis": 3}
        f.convergence = {"taille": 8.0, "kg_par_kN": f.mesures["kg_par_kN"] * (0.95 + 0.02 * rng.random())}
    s = c.statistiques()
    verifier("statistiques : part de L_eq et de eta sur 7 concepts (somme 1)",
             math.isclose(sum(s["variance"][k] for k in ("l_eq", "eta", "covariance")), 1.0), f"L_eq {s['variance']['l_eq']:.2f}")
    lien = next(x for x in s["liens"] if x["caracteristique"] == "n_faces")
    verifier("statistiques : lien parfait n_faces / kg/kN trouvé, p exact puis corrigé de Holm",
             math.isclose(lien["rho"], 1.0) and lien["p"] < 0.001 and lien["p_holm"] >= lien["p"], str(lien))
    verifier("statistiques : caractéristique constante écartée", all(x["caracteristique"] != "n_plis" for x in s["liens"]))
    m = s["maillage"]
    verifier("statistiques : bruit du maillage estimé (décalage 6 / 8 mm d'un concept à l'autre)",
             m["n"] == 7 and 0 < m["ecart_type_pct"] < 2 and 2 < m["decalage_moyen_pct"] < 6, str(m))
    verifier("statistiques : en phrases pour les IA", len(c.texte_statistiques()) >= 3)


def test_appliquer(dossier: Path) -> None:
    c = carnet_de_base(dossier)
    c.ajouter_lecon("Leçon existante")
    sortie = {
        "explication": "F001 fait mieux : chemin d'effort plus court.",
        "lecons": [
            {"id": "", "action": "nouvelle", "enonce": "Bonne leçon", "mecanisme": "m", "portee": "p",
             "comparaisons": [{"fiche_a": "F001", "fiche_b": "F002", "grandeur": "kg_par_kN", "attendu": "<",
                               "valeur_a": 0.001}]},  # un nombre écrit par l'IA : ignoré
            {"id": "", "action": "nouvelle", "enonce": "Leçon que tout contredit", "mecanisme": "", "portee": "",
             "comparaisons": [{"fiche_a": "F001", "fiche_b": "F002", "grandeur": "kg_par_kN", "attendu": ">"}]},
            {"id": "", "action": "nouvelle", "enonce": "Leçon sur des fiches inventées", "mecanisme": "", "portee": "",
             "comparaisons": [{"fiche_a": "F999", "fiche_b": "F001", "grandeur": "kg_par_kN", "attendu": "<"},
                              {"fiche_a": "F001", "fiche_b": "F002", "grandeur": "kg/kN", "attendu": "<"},
                              {"fiche_a": "F001", "fiche_b": "F002", "grandeur": "face:L.dessus:eta", "attendu": "~"}]},
            {"id": "L001", "action": "preciser", "enonce": "Énoncé précisé", "mecanisme": "", "portee": "",
             "comparaisons": [{"fiche_a": "F001", "fiche_b": "F002", "grandeur": "face:L.dessus:l_eq_mm", "attendu": "<"}]},
            {"id": "L042", "action": "appuyer", "enonce": "", "mecanisme": "", "portee": "", "comparaisons": []},
            {"id": "", "action": "inventer", "enonce": "x", "mecanisme": "", "portee": "", "comparaisons": []},
            "pas un objet",
        ],
        "experiences": [
            {"lecon": "N1", "parent": "F001", "modification": "doubler le dos", "grandeur": "kg_par_kN", "sens": "baisse"},
            {"lecon": "", "parent": "F404", "modification": "x", "grandeur": "kg_par_kN", "sens": "baisse"},
            {"lecon": "L001", "parent": "F001", "modification": " ", "grandeur": "kg_par_kN", "sens": "baisse"},
        ],
    }
    b = an.appliquer(c, sortie, "test")
    noms = [x.enonce for x in c.lecons]
    nouvelle = next(x for x in c.lecons if x.enonce == "Bonne leçon")
    verifier("analyste : leçon appuyée par les mesures acceptée, valeurs lues par Gabarit (pas celles de l'IA)",
             nouvelle.statut == "observee" and nouvelle.preuves[0].valeurs[0] == c.fiche("F001").mesures["kg_par_kN"])
    verifier("analyste : leçon nouvelle que toutes ses comparaisons contredisent refusée", "Leçon que tout contredit" not in noms)
    inventee = next(x for x in c.lecons if x.enonce == "Leçon sur des fiches inventées")
    verifier("analyste : fiche inventée, grandeur inconnue, sens invalide : comparaisons refusées",
             not inventee.preuves and inventee.statut == "hypothese" and sum(r.startswith("comparaison") for r in b.rejets) == 3)
    verifier("analyste : leçon précisée et appuyée par une métrique de face",
             c.lecon("L001").enonce == "Énoncé précisé" and c.lecon("L001").preuves[0].grandeur == "face:L.dessus:l_eq_mm")
    verifier("analyste : leçon inconnue, action inconnue refusées, leçons au-delà de la limite signalées",
             any("L042" in r for r in b.rejets) and any("inventer" in r for r in b.rejets)
             and any("au-delà de la limite de 6" in r for r in b.rejets))
    b3 = an.appliquer(carnet_de_base(dossier), {"lecons": ["pas un objet", {"action": "nouvelle", "enonce": " "}],
                                                 "experiences": [42], "explication": 3}, "t")
    verifier("analyste : éléments illisibles ou vides refusés sans planter",
             sum("illisible" in r for r in b3.rejets) == 2 and any("sans énoncé" in r for r in b3.rejets), str(b3.rejets))
    e = [x for x in c.experiences if x.statut == "a_faire"]
    verifier("analyste : expérience gardée, liée à la leçon nouvelle « N1 », les autres refusées",
             len(e) == 1 and e[0].lecon == nouvelle.id and e[0].modification == "doubler le dos", b.texte()[:200])
    c2 = carnet_de_base(dossier)
    b2 = an.appliquer(c2, {"explication": "", "lecons": [], "experiences": sortie["experiences"][:1] * 3}, "t", 1)
    verifier("analyste : limite d'expériences respectée", len(c2.experiences) == 1 and any("limite" in r for r in b2.rejets))
    texte = an.prompt_demande(c, ["F002"])
    verifier("analyste : la demande contient les fiches, la décomposition exacte et les leçons",
             "F001 / " in texte or "F002 / F001" in texte and "L_eq x" in texte and "Leçons existantes" in texte)
    verifier("analyste : lecture du JSON d'une réponse bavarde",
             an.lire_json('Voici.\n```json\n{"a": 1}\n```') == {"a": 1} and an.lire_json("rien") is None)


# ---------- faux serveurs ----------
def sse_texte(texte: str, stop: str = "end_turn") -> bytes:
    ev = [("message_start", {"type": "message_start", "message": {
              "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
              "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1200, "output_tokens": 1}}}),
          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
          ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": texte}}),
          ("content_block_stop", {"type": "content_block_stop", "index": 0}),
          ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                             "usage": {"output_tokens": 300}}),
          ("message_stop", {"type": "message_stop"})]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in ev).encode()


REPONSE = {"explication": "F002 perd sur L_eq.", "experiences": [],
           "lecons": [{"id": "", "action": "nouvelle", "enonce": "Leçon du serveur", "mecanisme": "", "portee": "",
                       "comparaisons": [{"fiche_a": "F001", "fiche_b": "F002", "grandeur": "l_eq_mm", "attendu": "<"}]}]}
requetes: list[tuple[str, dict]] = []


class Serveur(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        corps = json.loads(self.rfile.read(int(self.headers["content-length"])))
        requetes.append((self.path, corps))
        if self.path.endswith("/v1/messages"):  # API Claude, en flux
            data, type_ = sse_texte(json.dumps(REPONSE)), "text/event-stream"
        elif "response_format" in corps:  # modèle ouvert sans sortie structurée : refus, Gabarit réessaie sans
            data, type_ = b'{"error": "response_format non pris en charge"}', "application/json"
            self.send_response(400)
            self.send_header("content-type", type_)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        else:
            texte = "<think>je compare</think>Analyse :\n```json\n" + json.dumps(REPONSE) + "\n```"
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": texte}}],
                               "usage": {"prompt_tokens": 900, "completion_tokens": 150}}).encode()
            type_ = "application/json"
        self.send_response(200)
        self.send_header("content-type", type_)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_serveurs(dossier: Path) -> None:
    srv = HTTPServer(("127.0.0.1", 0), Serveur)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    c = carnet_de_base(dossier)
    client = anthropic.Anthropic(api_key="test", base_url=url, max_retries=0)
    b = an.analyser_claude(c, ["F002"], client=client, afficher=lambda *a, **k: None)
    corps = requetes[-1][1]
    verifier("analyste Claude : sortie structurée (schéma JSON), réflexion adaptative, modèle, flux",
             corps["output_config"]["format"]["schema"] == an.SCHEMA and corps["thinking"]["type"] == "adaptive"
             and corps["model"] == "claude-opus-5-5" and corps.get("stream") is True)
    verifier("analyste Claude : leçon rangée après contrôle, jetons comptés",
             any(x.enonce == "Leçon du serveur" and x.statut == "observee" for x in c.lecons) and b.jetons["entree"] == 1200)
    c = carnet_de_base(dossier)
    n = len(requetes)
    b = an.analyser_ouvert(c, ["F002"], url=f"{url}/v1", modele="qwen3:8b", afficher=lambda *a, **k: None)
    verifier("analyste ouvert : schéma refusé par le serveur, réponse lue dans le texte (raisonnement retiré)",
             len(requetes) == n + 2 and "response_format" in requetes[n][1] and "response_format" not in requetes[n + 1][1]
             and any(x.enonce == "Leçon du serveur" for x in c.lecons), b.texte()[:120])
    srv.shutdown()


def test_reprise_parent() -> None:
    d = json.loads(LV.read_text(encoding="utf-8"))
    p = parametres_depuis_valeurs(d, CDC, VALEURS)
    v = proposer_valeurs(d, CDC, optuna.trial.FixedTrial(p))
    ecart = {k: (v[k], VALEURS[k]) for k in VALEURS if not (v[k] == VALEURS[k] or abs(v[k] - VALEURS[k]) < 1e-6)}
    verifier("reprise : les paramètres tirés des valeurs du parent redonnent ces valeurs", not ecart, str(ecart))
    enfant = json.loads(json.dumps(d))
    enfant["variables"]["h_nervure"] = {"type": "reel", "min": 5, "max": 30}
    enfant["variables"]["t_L"] = {"type": "choix", "valeurs": [3, 4]}  # 5 mm n'est plus permis
    p2 = parametres_depuis_valeurs(enfant, CDC, VALEURS)
    verifier("reprise : variable nouvelle et valeur hors domaine laissées au tirage, le reste repris",
             "h_nervure" not in p2 and "t_L" not in p2 and p2["t_V"] == 5 and len(p2) == len(p) - 1)


# ---------- boucle avec doublures ----------
def kg_8mm(l_eq: float) -> float:
    return 0.954 * 1000 * RHO / RE * l_eq / 0.233


class Doublures:
    def __init__(self, dossier: Path, carnet: Carnet):
        self.dossier, self.carnet = dossier, carnet
        self.appels: list[dict] = []
        self.kg = {"ameliorer": (1700.0, 4.2), "tester": (1950.0, 4.4), "explorer": (2300.0, 4.0)}
        self.panne: Exception | None = None
        self.sans_concept = False

    def concepteur(self, reglages, consigne, lecons, exemples, calcul, dossier, afficher):
        op = "tester" if "Expérience E" in consigne else "ameliorer" if "Améliore" in consigne else "explorer"
        self.appels.append({"op": op, "consigne": consigne, "lecons": lecons, "exemples": exemples, "calcul": calcul})
        if self.panne is not None:
            raise self.panne
        s = Session(dossier)
        if self.sans_concept:
            s.arret = "limite de tours atteinte"
            return s
        d = json.loads(LV.read_text(encoding="utf-8"))
        d["nom"] = f"L + V {op} {len(self.appels)}"
        if op == "ameliorer":
            d["experience"] = {"parent": "F002", "modification": "retours sur la diagonale imposés",
                               "grandeur": "kg_par_kN", "sens": "baisse", "lecon": "L001"}
            d["prediction"] = {"kg_par_kN": 0.26, "mode_limite": "plastification", "principes": ["P3", "p7"]}
        reglages.sortie.mkdir(parents=True, exist_ok=True)
        chemin = reglages.sortie / f"concept_{len(self.appels)}.json"
        chemin.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        s.concepts.append(chemin)
        s.arret = "terminé"
        return s

    def optimiseur(self, chemin, reglages, depart, dossier, afficher):
        sur_disque = Carnet.charger(self.carnet.chemin)
        self.appels[-1].update(depart=depart, attente_sur_disque=[e.id for e in sur_disque.experiences_a_faire()],
                               predictions_sur_disque=[(p.id, p.statut) for p in sur_disque.predictions])
        l_eq, _ = self.kg[self.appels[-1]["op"]]  # kg/kN de l'optimiseur à 8 mm : 4,6 % sous celui de la fiche
        return {"valeurs": dict(VALEURS), "kg_par_kN": kg_8mm(l_eq), "faisable": True, "essais": 64, "faisables": 20,
                "budget": list(reglages.budget if depart else reglages.budget_exploration), "taille": 8.0}

    def mesureur(self, carnet, chemin, valeurs, reglages, dossier, **fiche):
        l_eq, masse = self.kg[self.appels[-1]["op"]]
        return carnet.ajouter_fiche(cr_factice(json.loads(Path(chemin).read_text())["nom"], l_eq, 0.233, masse),
                                    fichier=chemin, **fiche)

    def analyste(self, reglages, carnet, nouvelles, experiences_max, afficher):
        self.appels[-1]["place"] = experiences_max
        sortie = {"explication": f"analyse de {nouvelles}", "lecons": [], "experiences": [
            {"lecon": "", "parent": nouvelles[0], "modification": "écarter les vis", "grandeur": "flambement",
             "sens": "hausse"}] if len(carnet.experiences) < 2 else []}
        return an.appliquer(carnet, sortie, "doublure", experiences_max)


def test_boucle(dossier: Path) -> None:
    c = carnet_de_base(dossier)
    c.ajouter_lecon("Leçon à appliquer")
    c.enregistrer()
    reglages = bo.Reglages(sortie=dossier / "concepts", explorer_tous=3)
    dbl = Doublures(dossier, c)
    kw = {"concepteur": dbl.concepteur, "optimiseur": dbl.optimiseur, "mesureur": dbl.mesureur, "analyste": dbl.analyste,
          "afficher": lambda *a, **k: None}
    verifier("boucle : opération choisie (améliorer le meilleur quand rien n'est prévu)",
             bo.choisir_operation(c, 1)[0] == "ameliorer" and bo.choisir_operation(c, 3)[0] == "explorer"
             and bo.choisir_operation(Carnet(dossier / "vide.json"), 1)[0] == "explorer")
    bo.apprendre(3, reglages, c, dossier / "journal", **kw)
    ops = [a["op"] for a in dbl.appels]
    verifier("boucle : améliorer, puis tester l'expérience proposée par l'analyste, puis explorer",
             ops == ["ameliorer", "tester", "explorer"], str(ops))
    a1, a2, a3 = dbl.appels
    e1, e2 = c.experiences[0], c.experiences[1]
    verifier("boucle : prédiction de l'IA Concepteur enregistrée sur le disque avant l'optimisation, parent imposé",
             a1["attente_sur_disque"] == ["E001"] and e1.parent == "F001" and e1.lecon == "L001"
             and e1.modification == "retours sur la diagonale imposés")
    pr = c.predictions[0] if c.predictions else None
    verifier("boucle : prédiction physique sur le disque avant l'optimisation, puis jugée sur la fiche",
             a1["predictions_sur_disque"] == [("PR001", "a_verifier")] and len(c.predictions) == 1
             and pr.experience == "E001" and pr.principes == ["P3", "P7"] and pr.fiche == "F003"
             and pr.statut == "verifiee" and pr.verdict == "juste" and math.isclose(pr.mesure["ecart_pct"], 6.68, abs_tol=0.05),
             pr.texte() if pr else "aucune prédiction")
    verifier("boucle : pendant une expérience, ni calcul rapide ni exemple, concept du parent dans la consigne",
             not a1["calcul"] and a1["exemples"] == [] and "idée d'Iwan" in a1["consigne"]
             and "Garde les noms des variables" in a1["consigne"] and not a2["calcul"])
    verifier("boucle : un enfant part des cotes optimisées de son parent, un concept exploré de rien",
             a1["depart"] == [c.fiche("F001").valeurs] and a2["depart"] == [c.fiche("F003").valeurs] and a3["depart"] is None
             and a3["calcul"] and a3["exemples"] is None)
    f3, f4, f5 = c.fiche("F003"), c.fiche("F004"), c.fiche("F005")
    verifier("boucle : fiches liées (parent, expérience, départ, convergence à 8 mm)",
             f3.parent == "F001" and f3.experience == "E001" and f3.optimisation["depart"] == "F001"
             and f3.convergence == {"taille": 8.0, "kg_par_kN": kg_8mm(1700.0)}
             and f4.parent == "F003" and f4.experience == "E002" and f4.optimisation["depart"] == "F003"
             and f5.parent is None and f5.experience is None)
    verifier("boucle : verdicts (E001 kg/kN en baisse confirmée, E002 flambement en hausse non concluante)",
             e1.verdict == "confirmee" and e1.enfant == "F003" and e2.verdict == "non_concluante" and e2.enfant == "F004",
             f"{e1.texte()} | {e2.texte()}")
    verifier("boucle : consigne de test avec la modification et la prédiction déjà écrite",
             "écarter les vis" in a2["consigne"] and "Prédiction déjà enregistrée : flambement en hausse" in a2["consigne"])
    verifier("boucle : place laissée à l'analyste pour de nouvelles expériences", a1["place"] == 3, str(a1["place"]))
    verifier("boucle : carnet enregistré, cycles comptés, journal écrit",
             Carnet.charger(c.chemin).cycles == 3 and (dossier / "journal" / "journal.md").exists()
             and "Verdict" in (dossier / "journal" / "journal.md").read_text(encoding="utf-8"))
    verifier("boucle : exploration avec la liste des principes déjà mesurés", "Principes déjà mesurés" in a3["consigne"])

    # une expérience que l'IA ne parvient pas à construire est abandonnée, pas perdue
    e = c.ajouter_experience("F001", "nervure impossible", "kg_par_kN", "baisse")
    c.cycles = 3  # cycle 4 : tester
    dbl.sans_concept = True
    bo.apprendre(1, reglages, c, dossier / "journal", **kw)
    verifier("boucle : expérience sans concept abandonnée avec la raison", e.statut == "abandonnee"
             and "limite de tours" in e.mesure["note"], e.texte())
    # panne de l'IA : cycle perdu, carnet intact, arrêt après trois pannes de suite
    dbl.sans_concept = False
    dbl.panne = urllib.error.URLError("connexion refusée")
    n_appels, n_fiches = len(dbl.appels), len(c.fiches)
    bo.apprendre(5, reglages, c, dossier / "journal", **kw)
    verifier("boucle : IA injoignable, trois essais puis arrêt, aucune fiche ajoutée",
             len(dbl.appels) - n_appels == 3 and len(c.fiches) == n_fiches and c.cycles == 4)
    dbl.panne = ValueError("bogue")
    try:
        bo.apprendre(1, reglages, c, dossier / "journal", **kw)
        propage = False
    except ValueError:
        propage = True
    verifier("boucle : une autre erreur n'est pas maquillée en panne de l'IA", propage)

    p = bo.enregistrer_prediction(c, {"description": "d", "experience": {"grandeur": "kg", "sens": "monte"}}, "F001", "ia")
    verifier("prédiction illisible : kg/kN en baisse par défaut, et la source le dit",
             p.grandeur == "kg_par_kN" and p.sens == "baisse" and "défaut" in p.source and p.modification == "d")


def test_commande(dossier: Path) -> None:
    c = carnet_de_base(dossier)
    c.enregistrer()
    r = subprocess.run([sys.executable, str(RACINE / "exemples" / "apprendre.py"), "--afficher", "--carnet", str(c.chemin)],
                       capture_output=True, text=True, encoding="utf-8")
    verifier("commande : --afficher montre le carnet", r.returncode == 0 and "| F001 |" in r.stdout, r.stderr[-200:])
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    r = subprocess.run([sys.executable, str(RACINE / "exemples" / "apprendre.py"), "--cycles", "1", "--carnet",
                        str(c.chemin)], capture_output=True, text=True, encoding="utf-8", env=env)
    verifier("commande : sans clé Anthropic, message clair et rien de lancé", r.returncode == 1 and "ANTHROPIC_API_KEY" in r.stdout)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for test in (test_statistiques, test_zone_appui, lambda: test_carnet(tmp / "carnet"), lambda: test_statistiques_carnet(tmp),
                     lambda: test_appliquer(tmp / "appliquer"), lambda: test_serveurs(tmp / "serveurs"),
                     test_reprise_parent, lambda: test_boucle(tmp / "boucle"), lambda: test_commande(tmp / "commande")):
            test()
    for nom, ok, detail in resultats:
        print(f"{'OK   ' if ok else 'ÉCHEC'} {nom}" + (f"  ({detail})" if detail and not ok else ""))
    echecs = sum(not ok for _, ok, _ in resultats)
    print(f"\n{len(resultats) - echecs}/{len(resultats)} tests réussis")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
