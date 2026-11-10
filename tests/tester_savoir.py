"""Tests hors ligne du savoir physique (sans CalculiX, sans clé API, sans réseau) : calculs de prédimensionnement
recoupés avec les chiffres du document de réflexion et la fiche F001 du L + V, outil « predimensionner »,
prédictions physiques (lecture, verdict), carnet (enregistrement, calibration des principes), prompt et outils
de l'IA Concepteur, mode texte du modèle ouvert.

Lancer :  python tests/tester_savoir.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "tests"))

from tester_apprentissage import cr_factice  # compte rendu cohérent, sans calcul

from gabarit.apprentissage import boucle as bo
from gabarit.apprentissage.carnet import Carnet
from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.ia import concepteur as co
from gabarit.ia import ouvert as ou
from gabarit.savoir import calculs as ca
from gabarit.savoir import prediction as sp
from gabarit.savoir import principes as pr
from gabarit.savoir.outil import predimensionner

CDC = CahierDesCharges.charger(RACINE / "cahiers" / "sae51_equerre.json")
LV = RACINE / "concepts" / "sae51_L_V.json"
E, NU, RE, RHO = 210000.0, 0.3, 235.0, 7850.0
resultats: list[tuple[str, bool, str]] = []


def verifier(nom: str, ok: bool, detail: str = "") -> None:
    resultats.append((nom, bool(ok), detail))


def proche(a: float, b: float, rel: float = 2e-3) -> bool:
    return math.isclose(a, b, rel_tol=rel)


def test_calculs() -> None:
    t = ca.treillis(9810, 300, 133.4)
    verifier("treillis du L + V (h = 133,4 mm) : tirant 22,1 kN, bielle 24,1 kN, L_eq 1 483 mm, borne 0,0495 kg/kN",
             proche(t.tirant_N, 22061, 1e-3) and proche(t.bielle_N, 24144, 1e-3) and proche(t.l_eq, 1482.7, 1e-3)
             and proche(t.borne_kg_par_kN(RHO, RE), 0.04953, 1e-3), str(t))
    verifier("treillis : L_eq = (2L² + h²)/h, minimum 2 racine(2) L à h = L racine(2)",
             proche(ca.treillis(1, 300, ca.hauteur_optimale(300)).l_eq, 2 * math.sqrt(2) * 300, 1e-9)
             and proche(ca.treillis(1, 300, 180).l_eq, (2 * 300 ** 2 + 180 ** 2) / 180, 1e-12))
    s = ca.treillis(1, 300, 180, "symetrique")
    verifier("treillis symétrique : 8 % de moins à h = 180, minimum 2L à h = 2L",
             proche(s.l_eq, 1090, 1e-3) and proche(ca.treillis(1, 300, 600, "symetrique").l_eq, 600, 1e-9)
             and ca.hauteur_optimale(300, "symetrique") == 600)
    M = 9810 * 300
    verifier("tôle 150 x 5 : 4 709 MPa à plat, 157 MPa sur chant (b/t = 30 fois mieux)",
             proche(ca.contrainte_flexion_plaque(M, 150, 5), 4708.8) and proche(ca.contrainte_flexion_plaque(M, 5, 150), 156.96))
    u = ca.section_u(102, 20, 2)
    p = ca.section_plaque(102, 5)
    verifier("U 102 x 20 x 2 : 284 mm² (-44 %), inertie 7 389 mm⁴ (x7,0)",
             u.A == 284 and proche(u.I_faible, 7389, 2e-4) and proche(u.I_faible / p.I_faible, 6.95, 5e-3), f"{u.A} {u.I_faible}")
    verifier("raideur de plaque 189 800 MPa ; voilement de l'âme du U : 292 MPa",
             proche(ca.d_plaque(E, NU), 189800, 1e-4) and proche(ca.voilement(2, 102, 2, E, NU), 291.9, 1e-3))
    deux, un = ca.elancement_max(RE, 2), ca.elancement_max(RE, 1)
    verifier("b/t maxi d'une paroi comprimée en S235 : 42 et 14 (Eurocode), 57 et 19 (élastique)",
             deux["eurocode_classe_3"] == 42 and un["eurocode_classe_3"] == 14 and proche(deux["elastique"], 56.8, 2e-3)
             and proche(un["elastique"], 18.6, 3e-3))
    b = ca.barre_comprimee("diag", p, 319.76, 23514, E, NU, RE, RHO, K=0.575)
    verifier("diagonale plate du L + V : facteur de flambement 3,04 avec K = 0,575 (celui de la fiche F001)",
             proche(b.facteur_flambement, 3.04, 5e-3) and b.mode() == "flambement d'ensemble"
             and proche(b.masse_kg, 1.280, 1e-3), f"{b.facteur_flambement:.3f}")
    bu = ca.barre_comprimee("diag U", u, 319.76, 23514, E, NU, RE, RHO, K=0.575)
    verifier("diagonale en U de 2 mm : voilement de l'âme x3,53 (le mode), flambement d'ensemble x19, 0,71 kg",
             proche(bu.facteur_flambement, 3.53, 3e-3) and bu.mode() == "voilement de l'âme"
             and bu.facteur_euler > 19 and proche(bu.masse_kg, 0.713, 2e-3), f"{bu.facteur_flambement:.3f} {bu.mode()}")
    verifier("écrouissage EN 1993-1-3 : U de 2 mm 252,6 MPa, plafond 297,5 MPa",
             proche(ca.ecrouissage(235, 360, 2, 2, 284), 252.6, 1e-3) and ca.ecrouissage(235, 360, 5, 4, 100) == 297.5)
    verifier("soudure : f_vw,d 207,8 MPa, gorge 0,89 mm pour 37,9 kN sur deux cordons de 102 mm",
             proche(ca.fvw_d(360), 207.85, 1e-4) and proche(ca.gorge_min(37920, 102, 360), 0.894, 2e-3))
    verifier("vis M10 8.8 : 33,4 kN", proche(ca.vis_traction_rd(), 33408, 1e-9))
    verifier("identité kg/kN : L_eq 1 852 mm et eta 0,2236 donnent les 0,2767 kg/kN de F001",
             proche(ca.kg_par_kN(1852, 0.2236, RHO, RE), 0.2767, 1e-3))
    for args in ((9810, 300, 0), (9810, -1, 100)):
        try:
            ca.treillis(*args)
            ok = False
        except ValueError:
            ok = True
        verifier(f"treillis refusé pour {args}", ok)


def test_outil() -> None:
    r = predimensionner({"treillis": {"h": 133.4},
                         "bielles": [{"nom": "diag", "longueur": 319.76, "K": 0.575, "section": {"type": "U", "ame": 102, "aile": 20, "t": 2}}],
                         "parois": [{"nom": "aile", "b": 20, "t": 2, "bords": 1}, {"nom": "dessus", "b": 160, "t": 2, "bords": 1}],
                         "soudures": [{"nom": "diag", "effort_N": 37920, "longueur": 102}]}, CDC)
    b = r["bielles"][0]
    verifier("outil : treillis et bielle (effort pris dans le treillis, U, écrouissage)",
             r["treillis"]["bielle_N"] == 24144 and b["effort_N"] == 24144 and b["flambement_suffisant"]
             and b["mode"] == "voilement de l'âme" and b["re_ecroui_MPa"] == 252.6 and "a_faire" not in b, json.dumps(b)[:300])
    verifier("outil : paroi trapue et paroi élancée (avec la parade)",
             r["parois"][0]["trapue"] and not r["parois"][1]["trapue"] and "retour" in r["parois"][1]["a_faire"])
    verifier("outil : gorge minimale et retenue (3 mm)", r["soudures"][0]["gorge_retenue_mm"] == 3.0)
    plate = predimensionner({"treillis": {"h": 200}, "bielles": [{"longueur": 320, "section": {"type": "plaque", "b": 60, "t": 3}}]}, CDC)
    verifier("outil : bielle plate trop mince signalée (P3)",
             not plate["bielles"][0]["flambement_suffisant"] and "P3" in plate["bielles"][0]["a_faire"])
    haut = predimensionner({"treillis": {"h": 300}}, CDC)
    verifier("outil : h au-delà de la place disponible signalé", "alertes" in haut and "170" in haut["alertes"][0])
    mauvais = predimensionner({"treillis": {"h": -3}, "bielles": [{"longueur": 300, "section": {"type": "tube"}}, 5],
                               "parois": [{"b": "large", "t": 2}], "soudures": [{"longueur": 50}]}, CDC)
    verifier("outil : entrées fausses rendues comme erreurs lisibles, rien ne plante",
             len(mauvais["erreurs"]) == 5 and "treillis" not in mauvais, "; ".join(mauvais.get("erreurs", [])))
    verifier("outil : demande vide ou non objet", "aide" in predimensionner({}, CDC) and "erreurs" in predimensionner([], CDC))


def test_prediction() -> None:
    p, e = sp.lire({"prediction": {"kg_par_kN": "0.25", "mode_limite": "Flambement de la diagonale",
                                   "principes": ["p1", "P3", "P3"], "h_mm": 150, "raisonnement": "x" * 900}})
    verifier("prédiction lue et normalisée", e == [] and p["kg_par_kN"] == 0.25 and p["mode_limite"] == "flambement"
             and p["principes"] == ["P1", "P3"] and p["h_mm"] == 150 and len(p["raisonnement"]) == 600, str(e))
    p, e = sp.lire({"prediction": {"kg_par_kN": -1, "mode_limite": "rupture", "principes": "P12", "h_mm": "haut"}})
    verifier("prédiction : champs faux listés", len(e) == 4 and "kg_par_kN" not in p and "mode_limite" not in p, str(e))
    verifier("prédiction absente ou non objet", sp.lire({}) == (None, []) and sp.lire({"prediction": 3})[0] is None)
    verifier("mode tiré de la limite d'une fiche", [sp.mode_de_limite(x) for x in (
        "le flambement", "la traction dans les vis", "la plastification (contrainte au point critique), NON TENU (marge x0.95)",
        "la plastification (contrainte au point critique) (flambement non calculé)", "")]
             == ["flambement", "vis", "plastification", "plastification", ""])
    p, e = sp.lire({"prediction": {"kg_par_kN": 0.3, "mode_limite": "vis", "face_critique": " V.diag "}})
    v, m = sp.juger(0.25, "plastification", 0.27, "plastification", face_predite="V.diag", face_mesuree="V.jambe")
    verifier("lieu prédit : lu, jugé à part, sans effet sur le verdict",
             p["face_critique"] == "V.diag" and v == "juste" and m["lieu_juste"] is False and m["face_critique"] == "V.jambe"
             and "lieu_juste" not in sp.juger(0.25, "plastification", 0.27, "plastification")[1])
    verifier("verdicts : juste (écart et mode), partielle, fausse",
             [sp.juger(k, m, 0.27, "plastification")[0] for k, m in
              ((0.25, "plastification"), (0.35, "plastification"), (0.26, "flambement"), (0.40, "vis"))]
             == ["juste", "partielle", "partielle", "fausse"])


def test_carnet(dossier: Path) -> None:
    c = Carnet(dossier / "carnet.json")
    f1 = c.ajouter_fiche(cr_factice("L + V", 1852.0, 0.2236, 4.376), fichier=LV)
    f2 = c.ajouter_fiche(cr_factice("Variante", 1700.0, 0.25, 4.0), fichier=LV)
    f2.limite = "le flambement"
    f3 = c.ajouter_fiche(cr_factice("Cassé", 1800.0, 0.2, valide=False), fichier=LV)
    a = c.ajouter_prediction({"kg_par_kN": 0.25, "mode_limite": "plastification", "principes": ["P1", "P3"],
                              "face_critique": "L.dessus"}, LV, "L + V")
    b = c.ajouter_prediction({"kg_par_kN": 0.24, "mode_limite": "plastification", "principes": ["P3"]}, LV, "Variante",
                             erreurs=["« h_mm » : nombre attendu"])
    d = c.ajouter_prediction({"kg_par_kN": 0.2, "mode_limite": "vis"}, LV, "Cassé")
    verifier("carnet : prédictions numérotées PR001...", [x.id for x in c.predictions] == ["PR001", "PR002", "PR003"])
    c.conclure_prediction(a.id, f1.id)
    c.conclure_prediction(b.id, f2.id)
    c.conclure_prediction(d.id, f3.id)
    verifier("carnet : verdicts (juste à -10 %, partielle : bon kg/kN mais mauvais mode, non vérifiée sans mesures)",
             a.verdict == "juste" and math.isclose(a.mesure["ecart_pct"], -9.65, abs_tol=0.05) and a.mesure["mode_juste"] is True
             and b.verdict == "partielle" and b.mesure["mode_limite"] == "flambement" and b.mesure["valeur_juste"] is True
             and d.statut == "abandonnee" and "maillage" in d.mesure["note"], f"{a.texte()} | {b.texte()} | {d.texte()}")
    cal = c.calibration()
    verifier("calibration : par principe, justes, modes justes, biais",
             cal["toutes"]["n"] == 2 and cal["P3"]["n"] == 2 and cal["P3"]["justes"] == 1 and cal["P1"]["n"] == 1
             and cal["P3"]["modes_justes"] == 1 and math.isclose(cal["P1"]["ecart_moyen_pct"], -9.65, abs_tol=0.05)
             and cal["P3"]["lieux_predits"] == 1 and cal["P3"]["lieux_justes"] == 1 and a.mesure["lieu_juste"] is True
             and "lieu juste 1/1" in c.texte_calibration()[0] and "(lieu juste)" in a.texte(), str(cal))
    c.enregistrer()
    c2 = Carnet.charger(c.chemin)
    md = c.chemin.with_suffix(".md").read_text(encoding="utf-8")
    verifier("carnet : prédictions relues à l'identique, section et calibration dans le .md",
             [x.__dict__ for x in c2.predictions] == [x.__dict__ for x in c.predictions]
             and "## Prédictions physiques" in md and "P3 (Comprimé veut dire plié) : 2 prédiction(s)" in md
             and "3 prédiction(s) physique(s)" in md)
    prompt = c2.pour_prompt()
    verifier("carnet : calibration dans le prompt (complet et compact)",
             "Prédictions physiques vérifiées" in prompt and "PR001" in prompt
             and "Toutes : 2 prédiction(s)" in c2.pour_prompt(compact=True) and "PR001" not in c2.pour_prompt(compact=True))
    ancien = json.loads(c.chemin.read_text(encoding="utf-8"))
    ancien.pop("predictions")
    (dossier / "ancien.json").write_text(json.dumps(ancien), encoding="utf-8")
    verifier("carnet : un carnet d'avant les prédictions se charge", Carnet.charger(dossier / "ancien.json").predictions == [])
    try:
        c.ajouter_prediction({"mode_limite": "vis"}, LV)
        ok = False
    except ValueError:
        ok = True
    verifier("carnet : prédiction sans kg/kN refusée", ok)
    d = json.loads(LV.read_text(encoding="utf-8"))
    verifier("boucle : concept sans prédiction, rien d'enregistré",
             bo.enregistrer_prediction_physique(c, d, LV, "ia:x") is None)
    d["prediction"] = {"kg_par_kN": 0.24, "mode_limite": "flambement", "principes": ["P3", "P42"]}
    x = bo.enregistrer_prediction_physique(c, d, LV, "ia:x", "E001")
    verifier("boucle : prédiction du concept enregistrée, principe inconnu noté",
             x is not None and x.experience == "E001" and x.principes == ["P3"] and "P42" in x.erreurs[0]
             and x.concept == "concepts/sae51_L_V.json", x.texte() if x else "")
    d["prediction"] = {"kg_par_kN": 0.24}
    verifier("boucle : prédiction sans mode, pas enregistrée", bo.enregistrer_prediction_physique(c, d, LV, "ia:x") is None)


def test_ia(dossier: Path) -> None:
    prompt = co.prompt_systeme(CDC, [LV], "## Ce qu'on a appris\n- rien")
    court = co.prompt_systeme(CDC, [LV], "## Ce qu'on a appris\n- rien", savoir_compact=True)
    verifier("prompt : section savoir (neuf principes, chiffres du cas), champ prediction, méthode",
             "## Savoir physique" in prompt and all(f"- P{i}," in prompt for i in range(1, 10))
             and "h = 170 mm" in prompt and "Champ « prediction »" in prompt and "chiffre ton idée avec predimensionner" in prompt
             and "Formule" in prompt and "Formule" not in court and len(court) < len(prompt))
    verifier("prompt : le savoir est absent du cahier des charges (rien n'y est ajouté)",
             "Savoir" not in CDC.model_dump_json() and "principes" not in CDC.model_dump_json())
    verifier("outils : predimensionner déclaré", [o["name"] for o in co.OUTILS] == ["verifier_concept", "enregistrer_concept", "predimensionner"])
    session = co.Session(dossier)
    texte, err = co._executer_outil("predimensionner", {"treillis": {"h": 150}}, CDC, dossier, session, False)
    verifier("outils : predimensionner exécuté", not err and json.loads(texte)["treillis"]["h_mm"] == 150)
    texte, err = co._executer_outil("predimensionner", {"treillis": {"h": "?"}}, CDC, dossier, session, False)
    verifier("outils : predimensionner sans rien de calculable = erreur", err and "erreurs" in json.loads(texte))
    texte, err = co._executer_outil("predimensionner", "h=150", CDC, dossier, session, False)
    verifier("outils : predimensionner, entrée non objet", err)
    d = json.loads(LV.read_text(encoding="utf-8"))
    texte, _ = co._executer_outil("verifier_concept", {"concept": d}, CDC, dossier, session, False)
    verifier("verifier_concept : prédiction absente signalée", "prediction_a_corriger" in json.loads(texte))
    d["prediction"] = {"kg_par_kN": 0.25, "mode_limite": "plastification", "principes": ["P1"]}
    texte, _ = co._executer_outil("verifier_concept", {"concept": d}, CDC, dossier, session, False)
    verifier("verifier_concept : prédiction lisible, rien à corriger", "prediction_a_corriger" not in json.loads(texte))
    bloc = '```json\n{"predimensionner": {"treillis": {"h": 150}}}\n```'
    concept = '```json\n{"nom": "x", "toles": []}\n```'
    verifier("modèle ouvert, mode texte : demande de calcul reconnue, sauf si un concept la suit",
             ou._predimensionnement_dans_texte("Je chiffre :\n" + bloc) == {"treillis": {"h": 150}}
             and ou._predimensionnement_dans_texte(bloc + "\n" + concept) is None
             and ou._predimensionnement_dans_texte("pas de bloc") is None)
    verifier("modèle ouvert : consigne du mode texte et prompt compact",
             "predimensionner" in ou.CONSIGNE_TEXTE)


def test_principes() -> None:
    verifier("principes : neuf, numérotés P1 à P9, chacun avec règle, formule, validité et vérification",
             [p.id for p in pr.PRINCIPES] == [f"P{i}" for i in range(1, 10)]
             and all(p.regle and p.formule and p.validite and p.verifie for p in pr.PRINCIPES))
    h = pr.hauteur_disponible(CDC)
    verifier("hauteur disponible : de z = 10 à z = 180", h == {"z_bas_mm": 10.0, "z_vis_max_mm": 180.0, "h_mm": 170.0}, str(h))
    verifier("texte des principes pour un humain", pr.texte().count("## P") == 9)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for test in (test_calculs, test_outil, test_prediction, lambda: test_carnet(tmp), lambda: test_ia(tmp), test_principes):
            test()
    for nom, ok, detail in resultats:
        print(f"{'OK   ' if ok else 'ÉCHEC'} {nom}" + (f"  ({detail})" if detail and not ok else ""))
    echecs = sum(not ok for _, ok, _ in resultats)
    print(f"\n{len(resultats) - echecs}/{len(resultats)} tests réussis")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
