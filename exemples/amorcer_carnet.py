"""Amorce du carnet d'apprentissage (apprentissage/carnet.json) : ce que Gabarit sait avant le premier cycle.

1. Quatre fiches recalculées par la chaîne actuelle (compte rendu à 6 mm avec flambement, kg/kN à 8 mm pour le
   bruit du maillage) : L + V à l'optimum de la V1, et trois variantes à poche tirées de l'optimisation
   topologique (ESO) du dessus.
2. Les leçons de départ, chacune avec sa provenance : constats des phases 2 à 4 faits avant le carnet (notés comme
   tels), constats lus dans les fiches, comparaisons dont Gabarit lit lui-même les valeurs dans les fiches.
3. Une expérience pré-enregistrée sur la leçon la plus utile, que le premier cycle de la boucle réalisera
   (opération « tester ») : sa prédiction est écrite avant tout calcul de l'enfant.

Lancer depuis la racine du dépôt (environ 8 minutes) :
    python exemples/amorcer_carnet.py               # refuse d'écraser un carnet existant
    python exemples/amorcer_carnet.py --ecraser
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from gabarit.apprentissage.boucle import amorcer
from gabarit.apprentissage.carnet import (
    CARNET_DEFAUT,
    Carnet,
    Lecon,
    Preuve,
)

CONCEPTS = RACINE / "concepts"
VALEURS_LV = RACINE / "tests" / "donnees" / "sae51_L_V_optimum.json"
AVANT = "constat des phases 2 à 4, fait avant le carnet (chiffres non relus dans une fiche)"
SINGULIER = ("constat fait avant l'exclusion des extrémités de jonction : le pic lu à ces points dépendait surtout du "
             "maillage (135 à 183 MPa de 8 à 4 mm au bout du retour du dos de L + V)")


def observation(c: Carnet, x: Lecon, note: str, fiches=(), verdict: str = "conforme") -> None:
    """Constat écrit à la main (verdict « » : non jugé). Seules les expériences confirment ou contredisent."""
    x.preuves.append(Preuve("observation", list(fiches), verdict=verdict, note=note))
    c.mettre_a_jour_statut(x)


def fiches(c: Carnet, dossier: Path, comptes_rendus: Path) -> None:
    valeurs = json.loads(VALEURS_LV.read_text(encoding="utf-8"))["valeurs"]
    t0 = time.time()
    liste = [
        (CONCEPTS / "sae51_L_V.json", {"origine": "humain (idée d'Iwan)",
                                       "optimisation": {"essais": 168, "faisables": 46, "budget": [32, 96, 40],
                                                        "taille": 8.0}}),
        (CONCEPTS / "apprentissage" / "sae51_L_V_poche_eso.json", {"origine": "eso", "parent": "F001"}),
        (CONCEPTS / "apprentissage" / "sae51_L_V_poche_eso15.json", {"origine": "eso", "parent": "F001"}),
        (CONCEPTS / "apprentissage" / "sae51_L_V_poche_eso30.json", {"origine": "eso", "parent": "F001"}),
    ]
    for chemin, reglage in liste:
        ident = f"F{len(c.fiches) + 1:03d}"
        f = amorcer(c, chemin, valeurs, dossier=dossier / ident, **reglage)
        cr = dossier / ident / "compte_rendu.md"
        if cr.exists():
            shutil.copy(cr, comptes_rendus / f"{f.id}.md")
        print(f"{f.resume()}  ({time.time() - t0:.0f} s)", flush=True)


def lecons(c: Carnet) -> Lecon:
    """Leçons de départ ; renvoie celle sur la zone d'appui, qui porte l'expérience E001."""
    x = c.ajouter_lecon(
        "Une équerre en U simple (dos, dessus, deux ailes) plafonne vers 4 à 5 kN.",
        "Le dessus n'est porté que par ses bords : il plie localement sous l'appui, loin de tout raidisseur. Chemin "
        "d'effort en flexion, L_eq grande et eta faible.",
        "Pièces où l'effort arrive au milieu d'une tôle portée seulement par ses bords.", "humain (phase 3)")
    observation(c, x, AVANT + " : optimisation de l'équerre en U de la phase 3")

    x = c.ajouter_lecon(
        "Soutenir le dessus près de l'appui par une diagonale (L + V) fait passer l'effort en membrane : le L + V tient "
        "plus de trois fois l'effort de l'équerre en U.",
        "La diagonale descend l'effort en compression vers le bas du dos ; le dessus travaille en traction vers le haut "
        "du dos. Triangle dos, dessus, diagonale : chemin d'effort court, peu de flexion.",
        "Effort en porte-à-faux, fixation sur une face verticale.", "humain (phase 4)")
    observation(c, x, "fiche F001 : 15,8 kN tenus pour 4,38 kg, 22 % de l'énergie en flexion ; " + AVANT
                + " pour l'équerre en U", fiches=["F001"])

    x = c.ajouter_lecon(
        "Les contraintes se concentrent aux extrémités des lignes de soudure en T et aux jonctions en un seul point : "
        "préférer des liaisons sur toute la largeur et des retours qui se rejoignent bord à bord.",
        "Une liaison qui s'arrête au milieu d'une tôle concentre l'effort à son extrémité.",
        "Soudures en T, retours, doublages.", "humain (phase 4)")
    observation(c, x, SINGULIER + ". Gabarit exclut maintenant une boule de 10 mm autour de ces points et lit la "
                "contrainte à son bord : l'effet réel reste à mesurer par une expérience", verdict="")

    x = c.ajouter_lecon(
        "La bande du dos entre le pli et les vis du haut est fortement sollicitée (le dessus tire dessus) ; écarter les "
        "vis et doubler le dos aident.",
        "La traction du dessus passe par le pli dans le dos jusqu'aux vis du haut.", "Concepts en L vissés par le dos.",
        "humain (phase 4)")
    observation(c, x, SINGULIER + ". Avec la règle actuelle (fiche F001 à 6 mm), le dos culmine à 96 MPa près du pli "
                "contre 146 MPa au point critique, et 75 % de sa surface travaille sous 20 % de la contrainte utile : "
                "cette bande n'est pas ce qui limite L + V", fiches=["F001"], verdict="contraire")

    x = c.ajouter_lecon(
        "À l'optimum de L + V, la contrainte limite moins que le flambement de la diagonale (premier mode : V.diag "
        "100 %) : marge x1,6 en plastification, x1,01 au flambement. L'optimiseur a donc allégé jusqu'à ce que la "
        "diagonale soit sur le point de flamber.",
        "Diagonale de 5 mm comprimée, bords libres, sans retour (h_ret_diag = 0 à l'optimum).",
        "L + V et concepts dont une tôle comprimée porte l'effort.", "Gabarit (fiche F001)")
    observation(c, x, "fiche F001 à 6 mm : marges plastification 1,61, flambement 1,01, vis 3,05 ; mode V.diag 100 %, "
                "L.dessus 18 %", fiches=["F001"])

    x = c.ajouter_lecon(
        "Dans L + V, la jambe doublée contre le dos et le dos lui-même travaillent peu : 98 % et 75 % de leur surface "
        "sous 20 % de la contrainte utile, 22 % de la masse à eux deux.",
        "Le dos ne transmet l'effort qu'entre le pli du dessus, les vis et le pied de la jambe ; la jambe doublée "
        "double une matière déjà peu chargée.",
        "L + V ; à vérifier sur les concepts à doublage.", "Gabarit (fiche F001)")
    observation(c, x, "fiche F001 à 6 mm : V.jambe 9 % de la masse, eta 0,08 ; L.dos 13 % de la masse, eta 0,10. "
                "Les zones protégées (vis, plis, soudures, bande de bord) n'y laissent rien à évider pour l'ESO",
                fiches=["F001"])

    appui = c.ajouter_lecon(
        "Sur L + V, c'est la zone d'introduction de l'effort qui fixe l'effort tenu : 146 MPa au bout du dessus, à "
        "30 mm de l'axe de la charge, et autant dans la diagonale juste sous l'appui, contre 111 MPa au plus au-delà "
        "de 50 mm de cet axe (96 MPa dans le dos). Étaler l'effort sous l'appui est le premier levier ; alléger loin "
        "de l'appui ne paie que si cela ne fait pas monter cette zone.",
        "L'effort de 9,8 kN entre par le disque Ø50 de l'adaptateur, à 5 mm du bout libre du dessus et juste au-dessus "
        "de la soudure de la diagonale : le dessus fléchit localement autour de l'appui et la diagonale reçoit "
        "l'effort sur une courte longueur de soudure.",
        "Tous les concepts : chaque compte rendu dit à quelle distance de l'axe de la charge est le point critique et "
        "ce que vaut la contrainte au-delà de 50 mm.", "Gabarit (fiche F001)")
    observation(c, appui, "fiche F001 à 6 mm : point critique à 30 mm de l'axe de la charge, x1,31 sur la contrainte "
                "du reste de la pièce (diagnostic du compte rendu)", fiches=["F001"])
    observation(c, appui, "limite de la mesure : cette contrainte locale dépend de la modélisation de l'appui "
                "(efforts égaux sur les noeuds du disque Ø50, disque de 30 mm exclu) et de la maille. Optimum L + V : "
                "138, 146 puis 161 MPa à 8, 6 puis 4 mm (0,2625, 0,2767 puis 0,3057 kg/kN), le pic passant à 4 mm "
                "sur la jonction de la diagonale sous l'appui. Les fiches se comparent donc à maille égale",
                fiches=["F001"], verdict="")

    x = c.ajouter_lecon(
        "Sur L + V à son optimum, une poche au milieu du dessus ne fait pas baisser le kg/kN : à 4 % de la masse, "
        "l'écart reste dans le bruit car la contrainte utile monte autant que la masse baisse ; à 9 % de la masse, "
        "la pièce ne tient plus.",
        "Le milieu du dessus, peu contraint, transmet pourtant la traction du dessus vers le dos ; la poche la renvoie "
        "dans les bandes latérales et vers la zone d'appui, qui fixe l'effort tenu.",
        "Concepts limités par la zone d'appui ou par le flambement ; une poche ne paie que dans une tôle qui ne porte "
        "pas le chemin d'effort principal.", "Gabarit (fiches F002 à F004, optimisation topologique)")
    c.preuve_comparaison(x, "F001", "F002", "kg_par_kN", "<",
                         "poche de 45 cm² de la première ESO : écart sous le seuil, comme l'annonce la leçon")
    c.preuve_comparaison(x, "F001", "F003", "kg_par_kN", "<",
                         "poche de 46 cm² (ESO 15 % à 6 mm) : écart sous le seuil, comme l'annonce la leçon")
    c.preuve_comparaison(x, "F001", "F003", "sigma_utile", "<", "la poche de 46 cm² fait monter la contrainte utile")
    c.preuve_comparaison(x, "F001", "F004", "kg_par_kN", "<", "poche de 99 cm² (ESO 30 %) : pièce non tenue")
    return appui


def main() -> int:
    p = argparse.ArgumentParser(description="Amorce du carnet d'apprentissage")
    p.add_argument("--carnet", default=str(CARNET_DEFAUT))
    p.add_argument("--ecraser", action="store_true", help="remplacer un carnet existant")
    a = p.parse_args()
    chemin = Path(a.carnet)
    if chemin.exists() and not a.ecraser:
        print(f"{chemin} existe déjà : --ecraser pour le remplacer.")
        return 1
    comptes_rendus = chemin.parent / "comptes_rendus"
    comptes_rendus.mkdir(parents=True, exist_ok=True)
    c = Carnet(chemin)
    with tempfile.TemporaryDirectory(prefix="gabarit_amorce_") as tmp:
        fiches(c, Path(tmp), comptes_rendus)
    appui = lecons(c)
    c.ajouter_experience(
        "F001", "Prolonger le haut de la diagonale par une patte pliée à plat sous le dessus, centrée sous l'axe de la "
        "charge (x = 320) et soudée contre lui (doublage), pour étaler l'effort de l'appui sur deux épaisseurs et sur "
        "toute la largeur de la diagonale.", "kg_par_kN", "baisse", lecon=appui.id, source="Gabarit (amorce)")
    c.enregistrer()
    print(f"\nCarnet enregistré : {chemin} ({len(c.fiches)} fiches, {len(c.lecons)} leçons, "
          f"{len(c.experiences)} expérience)")
    print("\n".join(c.texte_statistiques()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
