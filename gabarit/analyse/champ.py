"""Champ mécanique par élément d'un calcul coque : contrainte, flexion ou membrane, énergie.

Source : sorties de CalculiX aux points d'intégration (calcul lancé avec details=True). Chaque coque S8R est
expansée en brique C3D20R : 4 points de Gauss dans le plan, sur deux niveaux de l'épaisseur
(zeta = -1/racine(3) pour les points 1 à 4, +1/racine(3) pour les points 5 à 8 ; vérifié sur une console).
La contrainte variant linéairement dans l'épaisseur (hypothèse des coques), en chaque point du plan :
    contrainte de membrane  s_m = (s_bas + s_haut) / 2         (au plan moyen)
    contrainte de flexion   s_f = racine(3) (s_haut - s_bas) / 2  (sa valeur sur les peaux)
    peaux                   s_m + s_f et s_m - s_f.
Pour toute forme quadratique Q (densité d'énergie), la moyenne dans l'épaisseur vaut exactement
Q(s_m) + Q(s_f) / 3 (les deux niveaux de Gauss intègrent un polynôme de degré 2) : on en tire la part de
l'énergie de déformation stockée en flexion. La Von Mises moyenne dans l'épaisseur, elle, n'est pas un
polynôme (racine d'une forme quadratique, avec un coude à la fibre neutre en flexion) : elle est intégrée
exactement, sous forme analytique, plutôt que lue aux deux points de Gauss (qui la surestimeraient de 15 % en
flexion pure).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RACINE3 = float(np.sqrt(3.0))


@dataclass
class ChampElements:
    ids: np.ndarray  # (n,) numéros d'éléments
    faces: list[list[str]]  # faces représentées par chaque élément (deux pour un doublage)
    centre: np.ndarray  # (n, 3) mm
    normale: np.ndarray  # (n, 3)
    aire: np.ndarray  # (n,) mm²
    t: np.ndarray  # (n,) mm, épaisseur du calcul (somme des tôles pour un doublage)
    vm_moyen: np.ndarray  # (n,) MPa, Von Mises moyen dans le volume (exact dans l'épaisseur, 4 points dans le plan)
    vm_peau: np.ndarray  # (n,) MPa, plus forte Von Mises sur les deux peaux (aux points de Gauss du plan)
    part_flexion: np.ndarray  # (n,) part de l'énergie de déformation due à la flexion (0 membrane pure, 1 flexion)
    energie: np.ndarray  # (n,) N.mm, énergie de déformation (CalculiX)
    poids: dict[str, np.ndarray]  # face -> part de chaque élément qui lui revient (t_face / t, 0 hors de la face)

    @property
    def volume(self) -> np.ndarray:
        return self.aire * self.t

    def poids_face(self, face: str) -> np.ndarray:
        return self.poids.get(face, np.zeros(len(self.ids)))


def von_mises(s: np.ndarray) -> np.ndarray:
    """Von Mises de tenseurs (..., 6) dans l'ordre sxx, syy, szz, sxy, sxz, syz."""
    sxx, syy, szz, sxy, sxz, syz = (s[..., k] for k in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy**2 + sxz**2 + syz**2))


def _q_vm(s: np.ndarray) -> np.ndarray:
    return von_mises(s) ** 2


def vm_moyenne_epaisseur(s_m: np.ndarray, s_f: np.ndarray) -> np.ndarray:
    """Moyenne exacte sur l'épaisseur de VM(s_m + zeta s_f), zeta de -1 à 1 (s_f : flexion en peau).
    VM² = c zeta² + 2 b zeta + a est une forme quadratique positive : avec x = zeta + b / c et
    k² = a / c - (b / c)² >= 0, VM = racine(c) racine(x² + k²), dont la primitive est
    (x racine(x² + k²) + k² asinh(x / k)) / 2."""
    a = _q_vm(s_m)
    c = _q_vm(s_f)
    b = 0.25 * (_q_vm(s_m + s_f) - _q_vm(s_m - s_f))
    membrane = c <= 1e-12 * np.maximum(a, 1e-30)
    cs = np.where(membrane, 1.0, c)
    x0 = b / cs
    k2 = np.maximum(a / cs - x0**2, 0.0)
    k = np.sqrt(k2)

    def primitive(x):
        r = np.sqrt(x**2 + k2)
        terme = np.where(k > 1e-12, k2 * np.arcsinh(x / np.where(k > 1e-12, k, 1.0)), 0.0)
        return 0.5 * (x * r + terme)

    integrale = np.sqrt(cs) * (primitive(1.0 + x0) - primitive(-1.0 + x0))
    return np.where(membrane, np.sqrt(a), 0.5 * integrale)


def densite_energie(s: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Densité d'énergie de déformation (N.mm/mm³) d'un matériau isotrope : [(1 + nu) s:s - nu (tr s)²] / 2E."""
    sxx, syy, szz, sxy, sxz, syz = (s[..., k] for k in range(6))
    ss = sxx**2 + syy**2 + szz**2 + 2 * (sxy**2 + sxz**2 + syz**2)
    tr = sxx + syy + szz
    return ((1 + nu) * ss - nu * tr**2) / (2 * E)


def champ_elements(modele, faces_elements: dict[int, list[str]], res,
                   epaisseur_face: dict[str, float] | None = None) -> ChampElements:
    """Champ par élément depuis un calcul statique lancé avec details=True. epaisseur_face : épaisseur propre de
    chaque face, pour partager un élément de doublage entre ses deux faces au prorata des épaisseurs."""
    if not res.contraintes_ip or not res.energie:
        raise ValueError("calcul sans sorties détaillées : relancer avec details=True")
    ids = np.array(sorted(modele.elements))
    coins = np.array([[modele.noeuds[n] for n in modele.elements[e][:4]] for e in ids])  # (n, 4, 3)
    d1, d2 = coins[:, 2] - coins[:, 0], coins[:, 3] - coins[:, 1]
    croix = np.cross(d1, d2)
    aire = 0.5 * np.linalg.norm(croix, axis=1)
    normale = croix / np.maximum(2 * aire, 1e-12)[:, None]
    centre = coins.mean(axis=1)
    t = np.array([(modele.epaisseurs or {}).get(int(e), modele.epaisseur) for e in ids])
    s = np.array([res.contraintes_ip[int(e)] for e in ids])  # (n, 8, 6)
    bas, haut = s[:, :4], s[:, 4:]
    s_m = 0.5 * (bas + haut)
    s_f = 0.5 * RACINE3 * (haut - bas)
    vm_peau = np.maximum(von_mises(s_m + s_f), von_mises(s_m - s_f)).max(axis=1)
    vm_moyen = vm_moyenne_epaisseur(s_m, s_f).mean(axis=1)
    E, nu = modele.materiau.E, modele.materiau.nu
    w_m = densite_energie(s_m, E, nu).sum(axis=1)
    w_f = densite_energie(s_f, E, nu).sum(axis=1) / 3.0
    part_flexion = w_f / np.maximum(w_m + w_f, 1e-30)
    energie = np.array([res.energie[int(e)] for e in ids])
    faces = [list(faces_elements.get(int(e), [])) for e in ids]
    poids: dict[str, np.ndarray] = {}
    for i, fs in enumerate(faces):
        tf = {f: (epaisseur_face or {}).get(f, 1.0) for f in fs}
        for f in fs:
            poids.setdefault(f, np.zeros(len(ids)))[i] = tf[f] / sum(tf.values())
    return ChampElements(ids, faces, centre, normale, aire, t, vm_moyen, vm_peau, part_flexion, energie, poids)
