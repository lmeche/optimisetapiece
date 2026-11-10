"""Moteur d'optimisation (phase 3).

1. Plan d'expériences quasi aléatoire (suite de Sobol, proche d'un Latin Hypercube) pour couvrir l'espace.
2. Optimisation bayésienne sous contraintes (Optuna, échantillonneur TPE) autour des meilleures zones.
3. Affinage local (COBYLA, sans dérivées, gère les contraintes) depuis le meilleur point faisable,
   variables continues seulement, les variables discrètes restant figées.

Les évaluations tournent en parallèle dans des processus séparés (Gmsh n'est pas utilisable en threads).
Une contrainte est respectée quand sa valeur est <= 0 ; un essai faisable respecte toutes les contraintes.
"""
from __future__ import annotations

import math
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
from scipy.optimize import minimize

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# proposer(trial) -> dict de paramètres ; evaluer(params) -> (objectif, contraintes, mesures)
Proposer = Callable[[optuna.Trial], dict]
Evaluateur = Callable[[dict], tuple[float, list[float], dict]]


@dataclass
class Bornes:
    """Variable continue affinable : nom du paramètre Optuna et ses bornes."""

    nom: str
    bas: float
    haut: float


def _contraintes(trial: optuna.trial.FrozenTrial) -> list[float]:
    return trial.user_attrs.get("contraintes", [1e9])


def faisable(trial: optuna.trial.FrozenTrial) -> bool:
    return trial.state == optuna.trial.TrialState.COMPLETE and all(c <= 0 for c in _contraintes(trial))


def meilleur_faisable(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    ok = [t for t in study.trials if faisable(t)]
    return min(ok, key=lambda t: t.value) if ok else None


def _executer(study, proposer, evaluer, lots, n_paralleles, journal):
    """Exécute des essais en parallèle. `lots` : liste de dicts de paramètres imposés (None = laisser le sampler)."""
    with ProcessPoolExecutor(max_workers=n_paralleles) as pool:
        attente = list(lots)
        en_cours = {}
        while attente or en_cours:
            while attente and len(en_cours) < n_paralleles:
                impose = attente.pop(0)
                if impose:
                    study.enqueue_trial(impose)
                trial = study.ask()
                params = proposer(trial)
                en_cours[pool.submit(evaluer, params)] = (trial, params)
            fini, _ = wait(en_cours, return_when=FIRST_COMPLETED)
            for f in fini:
                trial, params = en_cours.pop(f)
                try:
                    objectif, contraintes, mesures = f.result()
                except Exception as e:  # un calcul planté ne doit pas arrêter l'étude
                    trial.set_user_attr("erreur", str(e)[:500])
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    continue
                trial.set_user_attr("contraintes", [float(c) for c in contraintes])
                for i, c in enumerate(contraintes):
                    trial.set_constraint(f"c{i}", float(c))
                trial.set_user_attr("params_concept", params)
                for k, v in mesures.items():
                    trial.set_user_attr(k, v)
                study.tell(trial, objectif if math.isfinite(objectif) else 1e9)
                if journal:
                    journal(study.trials[trial.number])


def optimiser(proposer: Proposer, evaluer: Evaluateur, n_initial: int = 16, n_bayes: int = 48,
              affinage: list[Bornes] | None = None, n_affinage: int = 24, n_paralleles: int = 4,
              stockage: str | None = None, nom: str = "etude", graine: int = 0,
              journal: Callable[[optuna.trial.FrozenTrial], None] | None = None,
              depart: list[dict] | None = None) -> optuna.Study:
    """depart : paramètres imposés aux premiers essais du plan initial (point connu, optimum d'un parent) ;
    les paramètres qu'ils ne donnent pas sont tirés par l'échantillonneur."""
    study = optuna.create_study(study_name=nom, storage=stockage, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.QMCSampler(qmc_type="sobol", scramble=True, seed=graine))
    depart = list(depart or [])[:n_initial]
    _executer(study, proposer, evaluer, depart + [None] * (n_initial - len(depart)), n_paralleles, journal)

    study.sampler = optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True, seed=graine)
    _executer(study, proposer, evaluer, [None] * n_bayes, n_paralleles, journal)

    if affinage:
        _affiner(study, proposer, evaluer, affinage, n_affinage, n_paralleles, journal)
    return study


class _BudgetEpuise(Exception):
    pass


def _affiner(study, proposer, evaluer, variables: list[Bornes], budget: int, n_paralleles: int, journal):
    """COBYLA en variables normalisées [0, 1] depuis le meilleur point faisable ; chaque évaluation est un
    essai de l'étude (traçabilité). Les variables discrètes gardent la valeur du meilleur point.
    Le budget compte les calculs nouveaux : le point de départ, déjà calculé, n'est pas refait, et COBYLA est
    arrêté au budget même s'il en demande davantage (il exige au moins n + 2 évaluations)."""
    best = meilleur_faisable(study)
    if best is None or budget <= 0:
        return
    base = dict(best.params)
    cache: dict[tuple, optuna.trial.FrozenTrial] = {}
    u0 = np.array([(base[v.nom] - v.bas) / (v.haut - v.bas) for v in variables])
    if np.all((u0 >= 0) & (u0 <= 1)):
        cache[tuple(np.round(u0, 6))] = best
    nouveaux = 0

    def essai(u):
        nonlocal nouveaux
        cle = tuple(np.round(u, 6))
        if cle not in cache:
            if nouveaux >= budget:
                raise _BudgetEpuise
            nouveaux += 1
            p = dict(base)
            for v, ui in zip(variables, np.clip(u, 0, 1)):
                p[v.nom] = v.bas + float(ui) * (v.haut - v.bas)
            n = len(study.trials)
            _executer(study, proposer, evaluer, [p], 1, journal)
            cache[cle] = study.trials[n]
        return cache[cle]

    def valeur(u):
        t = essai(u)
        return t.value if t.state == optuna.trial.TrialState.COMPLETE else 1e9

    def contrainte(i):
        return lambda u: -_contraintes(essai(u))[i] if essai(u).state == optuna.trial.TrialState.COMPLETE else -1e9

    n_c = len(_contraintes(best))
    bornes = [{"type": "ineq", "fun": (lambda u, k=k: u[k])} for k in range(len(variables))]
    bornes += [{"type": "ineq", "fun": (lambda u, k=k: 1 - u[k])} for k in range(len(variables))]
    try:
        minimize(valeur, u0, method="COBYLA",
                 constraints=[{"type": "ineq", "fun": contrainte(i)} for i in range(n_c)] + bornes,
                 options={"rhobeg": 0.1, "maxiter": max(budget + 1, len(variables) + 2), "tol": 1e-3})
    except _BudgetEpuise:
        pass
