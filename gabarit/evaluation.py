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

    @property
    def respecte(self) -> bool:
        return self.valide and self.effort_tenu >= 9810.0


def evaluer(concept, cdc: CahierDesCharges, taille: float = 6.0, dossier: Path | None = None,
            flambement: bool = True, iterations_contact: int = 6) -> Resultat:
    erreurs = concept.verifier(cdc)
    if erreurs:
        return Resultat(valide=False, erreurs=erreurs)
    dossier = Path(dossier or tempfile.mkdtemp(prefix="gabarit_"))
    modele, info = concept.modele(cdc, taille)
    _contact(modele, info, dossier, iterations_contact)
    res = lancer(modele, dossier, "statique")

    vm = res.von_mises()
    co = res.coords
    xd = 20.0 + concept.t / 2
    def singulier(p):
        if abs(p[0] - xd) <= concept.t and any(np.hypot(p[1] - y, p[2] - z) <= 2 * 8.0 for y, z in info["trous"]):
            return True
        return np.hypot(p[0] - info["x_charge"], p[1]) <= 30.0 and p[2] > concept.z_dessus - concept.t
    sigma_max = max(vm.values())
    sigma_utile = max(v for n, v in vm.items() if not singulier(co[n]))
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
    if flambement:
        # Des modes parasites isolés apparaissent selon le maillage (constaté en phase 2) : on ne retient
        # que les facteurs retrouvés à 3 % près sur un second maillage 1,25 fois plus grossier.
        a = _facteurs(modele, dossier)
        modele2, info2 = concept.modele(cdc, 1.25 * taille)
        _contact(modele2, info2, dossier, iterations_contact, "contact2")
        b = _facteurs(modele2, dossier, "flambement2")
        confirmes = [x for x in a if any(abs(x - y) <= 0.03 * x for y in b)]
        facteur = min(confirmes) if confirmes else math.nan

    effort = ch.force * cdc.materiau.Re / sigma_utile
    masse = concept.masse(cdc.materiau.rho)
    return Resultat(True, [], masse, sigma_max, sigma_utile, fleche, facteur, effort,
                    masse / (effort / 1000), traction, len(modele.appuis_x), str(dossier))


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
