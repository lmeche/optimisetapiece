"""Statistiques du carnet : ce que les mesures disent vraiment, avec leur incertitude.

- Corrélation de rang de Spearman entre une caractéristique des concepts et leur kg/kN, avec une valeur p
  par permutation (exacte, toutes les permutations, jusqu'à 8 concepts ; sinon 20 000 tirages) : pas
  d'hypothèse de loi normale, valable pour quelques concepts seulement. Les valeurs p de plusieurs
  caractéristiques testées ensemble sont corrigées par la méthode de Holm (sinon, sur 15 caractéristiques, on
  trouverait presque toujours une « corrélation » au hasard).
- Décomposition de la variance de ln(kg/kN) entre concepts : kg/kN = 1000 rho/Re x L_eq / eta est une identité,
  donc ln(kg/kN) = cste + ln L_eq - ln eta, et la variance se partage exactement entre le chemin d'effort
  (L_eq), l'utilisation de la matière (eta) et leur covariance.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

PERMUTATIONS_EXACTES_MAX = 8
TIRAGES = 20000


def rangs(x) -> np.ndarray:
    """Rangs moyens (ex aequo partagés), comme scipy.stats.rankdata."""
    x = np.asarray(x, float)
    ordre = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[ordre] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y) -> float:
    rx, ry = rangs(x), rangs(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_p(x, y, graine: int = 0) -> tuple[float, float]:
    """Coefficient de Spearman et valeur p bilatérale par permutation (probabilité, si la caractéristique
    n'avait aucun lien avec y, d'observer une corrélation au moins aussi forte)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearman(x, y)
    if not math.isfinite(rho):
        return rho, math.nan
    rx, ry = rangs(x), rangs(y)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    obs = abs(float(rx @ ry)) - 1e-12
    n = len(x)
    if n <= PERMUTATIONS_EXACTES_MAX:
        perms = np.array(list(itertools.permutations(range(n))))
        stats = np.abs(ry[perms] @ rx)
        return rho, float(np.mean(stats >= obs))
    rng = np.random.default_rng(graine)
    stats = np.abs(np.array([ry[rng.permutation(n)] @ rx for _ in range(TIRAGES)]))
    return rho, float((1 + np.sum(stats >= obs)) / (TIRAGES + 1))


def holm(p: list[float]) -> list[float]:
    """Valeurs p corrigées de Holm-Bonferroni (contrôle du risque de trouver au moins un faux lien)."""
    ordre = sorted(range(len(p)), key=lambda i: p[i])
    m = len(p)
    corr = [math.nan] * m
    courant = 0.0
    for k, i in enumerate(ordre):
        if not math.isfinite(p[i]):
            continue
        courant = max(courant, min(1.0, (m - k) * p[i]))
        corr[i] = courant
    return corr


def decomposition_variance(kg_par_kN, l_eq, eta) -> dict[str, float]:
    """Parts de la variance de ln(kg/kN) entre concepts dues à L_eq, à eta et à leur covariance (somme = 1).
    Exact quand kg/kN = cste x L_eq / eta (même matériau)."""
    lk, ll, le = (np.log(np.asarray(v, float)) for v in (kg_par_kN, l_eq, eta))
    vk = float(np.var(lk))
    if len(lk) < 2 or vk <= 0:
        return {"n": len(lk), "variance": vk, "l_eq": math.nan, "eta": math.nan, "covariance": math.nan}
    vl, ve = float(np.var(ll)), float(np.var(le))
    c = float(np.mean((ll - ll.mean()) * (le - le.mean())))
    return {"n": len(lk), "variance": vk, "l_eq": vl / vk, "eta": ve / vk, "covariance": -2 * c / vk,
            "ecart_identite": float(np.max(np.abs((lk - lk.mean()) - ((ll - ll.mean()) - (le - le.mean())))))}
