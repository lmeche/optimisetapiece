"""Phase 4 : optimisation d'un concept au format déclaratif.

Lancer depuis la racine du dépôt :
    python exemples/optimiser_concept.py concepts/sae51_L_V.json [n_paralleles] [--rapport]
        [--budget initial,bayes,affinage]   (défaut 32,96,40)
Résultats dans resultats/<nom du fichier>/ : base Optuna, journal CSV, nuage des essais, croquis, synthèse.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402

from gabarit.concepts.format import ConceptJSON  # noqa: E402
from gabarit.etudes.concept_json import EtudeConcept  # noqa: E402
from gabarit.evaluation import evaluer  # noqa: E402
from gabarit.fem.calculix import Resultats  # noqa: E402
from gabarit.optimisation import faisable, meilleur_faisable, optimiser  # noqa: E402
from gabarit.rapport.image import image_von_mises  # noqa: E402

CDC = RACINE / "cahiers" / "sae51_equerre.json"


def main(chemin: Path, n_paralleles: int, rapport_seul: bool, budget: tuple[int, int, int] = (32, 96, 40),
         cdc: Path = CDC) -> None:
    etude = EtudeConcept(chemin, cdc)
    sortie = RACINE / "resultats" / chemin.stem
    sortie.mkdir(parents=True, exist_ok=True)
    stockage = f"sqlite:///{sortie / 'etude.db'}"
    t0 = time.time()
    if not rapport_seul:
        reprise = (sortie / "essais.csv").exists()  # une étude existante est poursuivie : on complète le journal
        with (sortie / "essais.csv").open("a" if reprise else "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not reprise:
                w.writerow(["numero", "faisable", "objectif_kg_par_kN", "masse_kg", "effort_tenu_N", "flambement",
                            "traction_vis", "valeurs"])
            ref = {}

            def journal(tr):
                w.writerow([tr.number, faisable(tr), tr.value] +
                           [tr.user_attrs.get(k) for k in ("masse_kg", "effort_tenu_N", "flambement", "traction_vis")] +
                           [json.dumps(tr.user_attrs.get("params_concept"), ensure_ascii=False)])
                f.flush()
                best = meilleur_faisable(ref["s"]) if "s" in ref else None
                print(f"[{time.time() - t0:6.0f} s] essai {tr.number:3d} faisable={faisable(tr)!s:5s} "
                      f"obj={tr.value:.4f} meilleur={best.value if best else float('nan'):.4f}", flush=True)

            ref["s"] = optuna.create_study(study_name=chemin.stem, storage=stockage, load_if_exists=True)
            optimiser(etude.proposer, etude.evaluer, n_initial=budget[0], n_bayes=budget[1],
                      affinage=etude.affinage(), n_affinage=budget[2], n_paralleles=n_paralleles,
                      stockage=stockage, nom=chemin.stem, journal=journal)
        print(f"Optimisation terminée en {time.time() - t0:.0f} s")
    study = optuna.load_study(study_name=chemin.stem, storage=stockage)
    rapport(study, etude, sortie, chemin)


def rapport(study, etude: EtudeConcept, sortie: Path, chemin: Path) -> None:
    best = meilleur_faisable(study)
    fichier = chemin.relative_to(RACINE) if chemin.is_relative_to(RACINE) else chemin
    synthese = {"concept": etude.definition["nom"], "fichier_concept": str(fichier), "essais": len(study.trials),
                "faisables": sum(faisable(t) for t in study.trials)}
    if best is None:
        ok = [t for t in study.trials if t.user_attrs.get("effort_tenu_N")]
        best = max(ok, key=lambda t: t.user_attrs["effort_tenu_N"])
        synthese.update({"meilleur_effort_tenu_N": best.user_attrs["effort_tenu_N"],
                         "valeurs": best.user_attrs["params_concept"]})
        trace_nuage(study, best, sortie / "essais.png", etude.definition["nom"], faisable_trouve=False)
    else:
        valeurs = best.user_attrs["params_concept"]
        verif = {}
        for taille in (6.0, 4.0):
            concept = ConceptJSON.charger(chemin, valeurs, etude.cdc)
            verif[f"{taille:g} mm"] = asdict(evaluer(concept, etude.cdc, taille=taille,
                                                     dossier=sortie / f"optimum_{taille:g}mm"))
        res = Resultats(sortie / "optimum_6mm" / "statique.frd", sortie / "optimum_6mm" / "statique.dat")
        image_von_mises(res, sortie / "optimum_von_mises.png", f"{etude.definition['nom']} optimisé, 9 810 N")
        trace_nuage(study, best, sortie / "essais.png", etude.definition["nom"])
        synthese.update({"meilleur_essai": best.number, "objectif_exploration_kg_par_kN": best.value,
                         "valeurs": valeurs, "geometrie": _geometrie(chemin, valeurs, etude), "verification": verif})
    (sortie / "synthese.json").write_text(json.dumps(synthese, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    print(json.dumps(synthese, indent=2, ensure_ascii=False, default=float))


def _geometrie(chemin, valeurs, etude) -> dict:
    c = ConceptJSON.charger(chemin, valeurs, etude.cdc)
    return {f.id: {"epaisseur": f.t, "points": f.points.round(2).tolist()} for f in c.geometrie().faces}


def trace_nuage(study, best, chemin: Path, titre: str, faisable_trouve: bool = True) -> None:
    pts = [(t.user_attrs.get("effort_tenu_N"), t.user_attrs.get("masse_kg"), faisable(t)) for t in study.trials
           if t.user_attrs.get("masse_kg") is not None and t.user_attrs.get("effort_tenu_N") is not None]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ok, coul, lab in ((False, "#bbbbbb", "non faisable"), (True, "#1f77b4", "faisable")):
        xs = [p[0] / 1000 for p in pts if p[2] == ok and math.isfinite(p[0])]
        ys = [p[1] for p in pts if p[2] == ok and math.isfinite(p[0])]
        ax.scatter(xs, ys, s=18, c=coul, label=lab)
    lab = (f"optimum : {best.value:.3f} kg/kN" if faisable_trouve
           else f"plus grand effort tenu : {best.user_attrs['effort_tenu_N'] / 1000:.1f} kN")
    ax.scatter([best.user_attrs["effort_tenu_N"] / 1000], [best.user_attrs["masse_kg"]], s=140, marker="*",
               c="#d62728", label=lab, zorder=5)
    ax.axvline(9.81, color="k", ls="--", lw=1)
    ax.set_xlabel("Effort tenu avant plastification (kN)")
    ax.set_ylabel("Masse (kg)")
    ax.set_title(f"{titre} : essais de l'optimisation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Optimisation d'un concept au format déclaratif")
    p.add_argument("concept")
    p.add_argument("n_paralleles", nargs="?", type=int, default=4)
    p.add_argument("--rapport", action="store_true", help="refaire seulement le rapport d'une étude existante")
    p.add_argument("--budget", default="32,96,40", help="essais initiaux, bayésiens, d'affinage")
    p.add_argument("--cdc", default=str(CDC), help="cahier des charges (JSON)")
    a = p.parse_args()
    main(Path(a.concept).resolve(), a.n_paralleles, a.rapport, tuple(int(x) for x in a.budget.split(",")),
         Path(a.cdc).resolve())
