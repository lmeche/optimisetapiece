"""Évaluation d'un concept : statique (avec contact unilatéral dos / socle) + flambement linéaire."""
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.fem.calculix import lancer

PRECHARGE_LIMITE_M10_88 = 0.7 * 800 * 58.0  # N, 70 % de Rm x section résistante (ordre de grandeur)
RAYON_VIS_SINGULIER = 16.0  # mm, anneau exclu autour de chaque vis, sur la face de fixation
RAYON_APPUI_SINGULIER = 30.0  # mm, disque exclu sous l'appui (effort réparti sur peu de noeuds)
RAYON_JONCTION_SINGULIER = 10.0  # mm, boule exclue autour de chaque extrémité de jonction (voir points_singuliers)


@dataclass
class Resultat:
    valide: bool
    erreurs: list[str] = field(default_factory=list)
    masse_kg: float = math.nan
    sigma_max: float = math.nan  # MPa, tous noeuds
    sigma_max_utile: float = math.nan  # MPa, hors singularités (bord des vis, sous l'appui)
    fleche: float = math.nan  # mm, sous l'appui
    facteur_flambement: float = math.nan
    effort_tenu: float = math.nan  # N, première plastification (linéaire, sur sigma_max_utile)
    masse_par_kN: float = math.nan  # kg/kN tenu
    traction_vis_max: float = math.nan  # N
    noeuds_contact: int = 0
    dossier: str = ""
    point_sigma_utile: tuple[float, float, float] | None = None  # où se trouve sigma_max_utile (mm)

    @property
    def respecte(self) -> bool:
        return self.valide and self.effort_tenu >= 9810.0


def evaluer(concept, cdc: CahierDesCharges, taille: float = 6.0, dossier: Path | None = None,
            flambement: bool = True, iterations_contact: int = 6, garde: dict | None = None) -> Resultat:
    """garde : dictionnaire à remplir pour l'analyse détaillée (gabarit/analyse) : modèle, résultats statiques
    avec contraintes aux points d'intégration et énergie par élément, déformées des modes de flambement."""
    erreurs = concept.verifier(cdc)
    if erreurs:
        return Resultat(valide=False, erreurs=erreurs)
    dossier = Path(dossier or tempfile.mkdtemp(prefix="gabarit_"))
    details = garde is not None
    modele, info = concept.modele(cdc, taille)
    _contact(modele, info, dossier, iterations_contact)
    res = lancer(modele, dossier, "statique", details=details)

    vm = res.von_mises()
    co = res.coords
    sigma_max = max(vm.values())
    n_pic, sigma_utile = pic_utile(res, info)
    ch = cdc.chargements[0]
    fleche = -float(np.mean([res.u_noeuds[n][2] for n in info["noeuds_charge"]]))

    # Traction par vis : somme des réactions X des noeuds serrés autour de chaque trou
    rf = res.reactions.get("ENCASTRE", {})
    traction = 0.0
    for y, z in info["trous"]:
        fx = sum(f[0] for n, f in rf.items()
                 if np.hypot(modele.noeuds[n][1] - y, modele.noeuds[n][2] - z) <= 8.0 + 1e-6)
        traction = max(traction, -fx)

    facteur = math.nan
    if details:  # faces de chaque élément copiées : le second maillage du flambement les remplace dans le concept
        garde.update(modele=modele, info=info, statique=res, n_pic=n_pic, dossier=dossier,
                     faces_elements=dict(getattr(concept, "faces_elements", {})))
    if flambement:
        # Des modes parasites isolés apparaissent selon le maillage (constaté en phase 2) : on ne retient
        # que les facteurs retrouvés à 3 % près sur un second maillage 1,25 fois plus grossier.
        rb = lancer(modele, dossier, "flambement", analyse="flambement", n_modes=8, fichiers=False, details=details)
        a = [f for f in rb.facteurs_flambement if f > 0]
        modele2, info2 = concept.modele(cdc, 1.25 * taille)
        _contact(modele2, info2, dossier, iterations_contact, "contact2")
        b = _facteurs(modele2, dossier, "flambement2")
        confirmes = [x for x in a if any(abs(x - y) <= 0.03 * x for y in b)]
        facteur = min(confirmes) if confirmes else math.nan
        if details:
            garde.update(flambement=rb, facteurs_maillage1=a, facteurs_maillage2=b, facteur_confirme=facteur)

    effort = ch.force * cdc.materiau.Re / sigma_utile
    masse = concept.masse(cdc.materiau.rho)
    return Resultat(True, [], masse, sigma_max, sigma_utile, fleche, facteur, effort,
                    masse / (effort / 1000), traction, len(modele.appuis_x), str(dossier),
                    tuple(round(float(c), 1) for c in co[n_pic]))


def pic_utile(res, info: dict) -> tuple[int, float]:
    """Noeud et valeur de la plus forte contrainte de Von Mises hors singularités : anneaux de 16 mm autour des
    vis sur la face de fixation, disque de 30 mm sous l'appui (charge ponctuelle répartie sur peu de noeuds),
    boules de 10 mm autour des extrémités de jonction (info["singuliers"]). La contrainte au bord de ces zones
    compte toujours : une jonction qui s'arrête dans une zone très chargée reste pénalisée, par la contrainte lue
    à 10 mm de son extrémité plutôt qu'au point singulier, dont la valeur ne dépend que du maillage."""
    vm = res.von_mises()
    co = res.coords
    n_pic = max((n for n in vm if not singulier(co[n], info)), key=vm.get)
    return n_pic, vm[n_pic]


def singulier(p, info: dict) -> bool:
    """Le point p est-il dans une zone exclue du critère de contrainte (voir `pic_utile`) ?"""
    xd, td = info["x_dos"], info["t_dos"]
    if abs(p[0] - xd) <= td and any(np.hypot(p[1] - y, p[2] - z) <= RAYON_VIS_SINGULIER for y, z in info["trous"]):
        return True
    if np.hypot(p[0] - info["x_charge"], p[1]) <= RAYON_APPUI_SINGULIER and abs(p[2] - info["z_charge"]) <= info["t_charge"]:
        return True
    pts = info.get("singuliers")
    return bool(pts) and float(np.min(np.linalg.norm(np.asarray(pts) - np.asarray(p), axis=1))) <= RAYON_JONCTION_SINGULIER


def _contact(modele, info, dossier: Path, iterations: int, nom: str = "contact") -> None:
    """Contact unilatéral du dos sur le socle par ensembles actifs : on bloque en X les noeuds du dos
    qui pénètrent (Ux < 0) et on relâche ceux dont la réaction tire (Fx < 0)."""
    for _ in range(iterations):
        res = lancer(modele, dossier, nom, fichiers=False)
        entrent = {n for n in info["dos"] if res.u_noeuds[n][0] < -1e-6}
        tirent = {n for n, f in res.reactions.get("APPUI", {}).items() if f[0] < 0}
        nouveau = (modele.appuis_x | entrent) - tirent
        if nouveau == modele.appuis_x:
            return
        modele.appuis_x = nouveau


def _facteurs(modele, dossier: Path, nom: str = "flambement", n_modes: int = 8) -> list[float]:
    rb = lancer(modele, dossier, nom, analyse="flambement", n_modes=n_modes, fichiers=False)
    return [f for f in rb.facteurs_flambement if f > 0]
