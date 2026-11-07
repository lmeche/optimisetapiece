"""Phase 3 : optimisation de l'équerre en U sur la SAE 5.1.

Lancer depuis la racine du dépôt :  python exemples/phase3_optimisation.py [n_paralleles]
Résultats dans resultats/phase3/ : base Optuna (SQLite), journal CSV, front de Pareto, croquis, synthèse JSON.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gabarit.cahier_des_charges import CahierDesCharges  # noqa: E402
from gabarit.concepts.equerre_u import EquerreU  # noqa: E402
from gabarit.etudes import sae51_equerre_u as etude  # noqa: E402
from gabarit.evaluation import evaluer  # noqa: E402
from gabarit.fem.calculix import Resultats  # noqa: E402
from gabarit.optimisation import faisable, meilleur_faisable, optimiser  # noqa: E402
from gabarit.rapport.image import image_von_mises  # noqa: E402

SORTIE = Path(__file__).resolve().parents[1] / "resultats" / "phase3"
CHAMPS = ["numero", "faisable", "objectif_kg_par_kN", "masse_kg", "effort_tenu_N", "flambement", "traction_vis",
          "t", "b", "z_dessus", "z_bas", "h_aile_dos", "h_aile_bout", "colonnes_vis"]


def main(n_paralleles: int = 4) -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    journal_csv = SORTIE / "essais.csv"
    t0 = time.time()
    with journal_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CHAMPS)
        w.writeheader()

        def journal(tr):
            p = tr.user_attrs.get("params_concept", {})
            ligne = {"numero": tr.number, "faisable": faisable(tr), "objectif_kg_par_kN": tr.value,
                     **{k: tr.user_attrs.get(k) for k in ("masse_kg", "effort_tenu_N", "flambement", "traction_vis")},
                     **{k: p.get(k) for k in CHAMPS[7:]}}
            w.writerow(ligne)
            f.flush()
            best = meilleur_faisable(study_ref[0]) if study_ref else None
            print(f"[{time.time() - t0:6.0f} s] essai {tr.number:3d} faisable={faisable(tr)!s:5s} "
                  f"obj={tr.value:.4f} meilleur={best.value if best else float('nan'):.4f}", flush=True)

        study_ref = []
        import optuna
        study_ref.append(optuna.create_study(study_name="sae51_equerre_u", storage=f"sqlite:///{SORTIE / 'etude.db'}",
                                             load_if_exists=True))
        study = optimiser(etude.proposer, etude.evaluer_params, n_initial=24, n_bayes=72,
                          affinage=etude.AFFINAGE, n_affinage=30, n_paralleles=n_paralleles,
                          stockage=f"sqlite:///{SORTIE / 'etude.db'}", nom="sae51_equerre_u", journal=journal)
    print(f"Optimisation terminée en {time.time() - t0:.0f} s, {len(study.trials)} essais")

    rapport(study, t0)


def rapport(study, t0: float | None = None) -> None:
    best = meilleur_faisable(study)
    if best is None:
        ok = [t for t in study.trials if t.user_attrs.get("effort_tenu_N")]
        best = max(ok, key=lambda t: t.user_attrs["effort_tenu_N"])
        trace_pareto(study, best, SORTIE / "pareto.png", faisable_trouve=False)
        synthese = {"essais": len(study.trials), "faisables": 0, "meilleur_effort_tenu_N": best.user_attrs["effort_tenu_N"],
                    "masse_kg": best.user_attrs["masse_kg"], "contraintes": best.user_attrs["contraintes"],
                    "parametres": best.user_attrs["params_concept"]}
        (SORTIE / "synthese.json").write_text(json.dumps(synthese, indent=2, ensure_ascii=False, default=float),
                                              encoding="utf-8")
        print("Aucun essai faisable. Plus grand effort tenu :")
        print(json.dumps(synthese, indent=2, ensure_ascii=False, default=float))
        return
    t0 = t0 or time.time()
    params = best.user_attrs["params_concept"]
    params["colonnes_vis"] = tuple(params["colonnes_vis"])
    cdc = CahierDesCharges.charger(etude.CDC)
    verif = {}
    for taille in (6.0, 4.0):  # vérification de l'optimum sur maillages plus fins
        r = evaluer(EquerreU(**params), cdc, taille=taille, dossier=SORTIE / f"optimum_{taille:g}mm")
        verif[f"{taille:g} mm"] = asdict(r)
    res = Resultats(SORTIE / "optimum_6mm" / "statique.frd", SORTIE / "optimum_6mm" / "statique.dat")
    image_von_mises(res, SORTIE / "optimum_von_mises.png", "Équerre en U optimisée, 9 810 N")
    trace_pareto(study, best, SORTIE / "pareto.png")
    synthese = {"essais": len(study.trials), "faisables": sum(faisable(t) for t in study.trials),
                "duree_s": round(time.time() - t0), "meilleur_essai": best.number,
                "objectif_exploration": best.value, "parametres": params, "verification": verif}
    (SORTIE / "synthese.json").write_text(json.dumps(synthese, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    print(json.dumps(synthese, indent=2, ensure_ascii=False, default=float))


def trace_pareto(study, best, chemin: Path, faisable_trouve: bool = True) -> None:
    pts = [(t.user_attrs.get("effort_tenu_N"), t.user_attrs.get("masse_kg"), faisable(t)) for t in study.trials
           if t.user_attrs.get("masse_kg") is not None and t.user_attrs.get("effort_tenu_N") is not None]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ok, coul, lab in ((False, "#bbbbbb", "non faisable"), (True, "#1f77b4", "faisable")):
        xs = [p[0] / 1000 for p in pts if p[2] == ok and math.isfinite(p[0])]
        ys = [p[1] for p in pts if p[2] == ok and math.isfinite(p[0])]
        ax.scatter(xs, ys, s=18, c=coul, label=lab)
    lab = (f"optimum : {best.value:.3f} kg/kN" if faisable_trouve
           else f"plus grand effort tenu : {best.user_attrs['effort_tenu_N'] / 1000:.1f} kN")
    ax.scatter([best.user_attrs["effort_tenu_N"] / 1000], [best.user_attrs["masse_kg"]], s=120, marker="*",
               c="#d62728", label=lab, zorder=5)
    ax.axvline(9.81, color="k", ls="--", lw=1)
    ax.text(9.9, ax.get_ylim()[1] * 0.95, "1 t", va="top")
    ax.set_xlabel("Effort tenu avant plastification (kN)")
    ax.set_ylabel("Masse (kg)")
    ax.set_title("Équerre en U : essais de l'optimisation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1:] == ["--rapport"]:  # régénère les sorties depuis la base existante
        import optuna
        rapport(optuna.load_study(study_name="sae51_equerre_u", storage=f"sqlite:///{SORTIE / 'etude.db'}"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
