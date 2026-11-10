"""Phase 3 : validation du moteur d'optimisation sur un cas dont l'optimum se calcule à la main.

Poutre console de section rectangulaire b x h soumise à un moment M = 2,943e6 N.mm (9 810 N x 300 mm).
Minimiser l'aire A = b.h sous la contrainte sigma = 6M / (b.h²) <= 235 MPa, b dans [10, 50], h dans [10, 60].
Optimum analytique : h = 60 (borne), b = 6M / (235 x 60²) = 20,87 mm, A = 1252,3 mm².
Critère : aire trouvée à moins de 2 % de l'optimum, contrainte respectée.
Second contrôle : l'affinage respecte son budget de calculs nouveaux, même inférieur au minimum de COBYLA (n + 2),
et ne recalcule pas son point de départ.
Lancer :  python tests/valider_optimiseur.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gabarit.optimisation import Bornes, meilleur_faisable, optimiser  # noqa: E402

M, SIG = 9810.0 * 300.0, 235.0


def proposer(trial):
    return {"b": trial.suggest_float("b", 10, 50), "h": trial.suggest_float("h", 10, 60)}


def evaluer(p):
    sigma = 6 * M / (p["b"] * p["h"] ** 2)
    return p["b"] * p["h"], [sigma / SIG - 1], {"sigma": sigma}


def main() -> int:
    a_theo = 6 * M / (SIG * 60**2) * 60
    study = optimiser(proposer, evaluer, n_initial=16, n_bayes=48,
                      affinage=[Bornes("b", 10, 50), Bornes("h", 10, 60)], n_affinage=40, n_paralleles=4)
    best = meilleur_faisable(study)
    ecart = 100 * (best.value - a_theo) / a_theo
    print(f"{len(study.trials)} essais ; meilleur : b = {best.params['b']:.2f}, h = {best.params['h']:.2f}, "
          f"A = {best.value:.1f} mm² (théorie {a_theo:.1f}), écart {ecart:+.2f} %, "
          f"sigma = {best.user_attrs['sigma']:.1f} MPa")
    ok = 0 <= ecart < 2
    petite = optimiser(proposer, evaluer, n_initial=4, n_bayes=4, affinage=[Bornes("b", 10, 50), Bornes("h", 10, 60)],
                       n_affinage=1, n_paralleles=2, graine=1)
    vus = [tuple(round(v, 9) for v in t.params.values()) for t in petite.trials]
    budget_ok = len(petite.trials) == 9 and len(set(vus)) == len(vus)
    print(f"Affinage au budget 1 : {len(petite.trials)} essais (attendu 4 + 4 + 1), "
          f"{len(vus) - len(set(vus))} doublon(s)")
    ok = ok and budget_ok
    print("Validation", "RÉUSSIE" if ok else "ÉCHOUÉE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
