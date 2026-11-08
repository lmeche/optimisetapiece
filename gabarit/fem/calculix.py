"""Interface minimale vers CalculiX : écriture du .inp, lancement de ccx, lecture des résultats.

Le modèle est décrit en coques (S8R, 8 noeuds) : c'est la représentation retenue pour les tôles
de l'ensemble mécano-soudé. Le format est volontairement simple pour être généré par programme.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Materiau:
    nom: str
    E: float  # MPa
    nu: float
    rho: float = 7.85e-9  # t/mm3 (acier)
    Re: float = 235.0  # MPa, limite élastique


@dataclass
class ModeleCoque:
    """Maillage coque + données du calcul (unités : mm, N, MPa, t)."""

    noeuds: dict[int, tuple[float, float, float]]
    elements: dict[int, list[int]]  # S8R : 8 noeuds par élément
    epaisseur: float
    materiau: Materiau
    encastres: set[int] = field(default_factory=set)  # noeuds bloqués en 6 ddl
    forces: dict[int, tuple[float, float, float]] = field(default_factory=dict)  # noeud -> (Fx, Fy, Fz)
    appuis_x: set[int] = field(default_factory=set)  # noeuds bloqués en X seulement (contact actif)
    epaisseurs: dict[int, float] | None = None  # épaisseur par élément (sinon `epaisseur` partout)


def trouver_ccx() -> str:
    for nom in ("ccx", "ccx_2.22", "ccx_2.21", "ccx.exe", "ccx_static.exe"):
        chemin = shutil.which(nom)
        if chemin:
            return chemin
    env = os.environ.get("GABARIT_CCX")
    if env and Path(env).exists():
        return env
    raise FileNotFoundError("CalculiX (ccx) introuvable : l'ajouter au PATH ou définir GABARIT_CCX.")


def ecrire_inp(modele: ModeleCoque, chemin: Path, analyse: str = "statique", n_modes: int = 3,
               fichiers: bool = True) -> None:
    m = modele
    lignes = ["*HEADING", "Gabarit", "*NODE, NSET=NALL"]
    lignes += [f"{i}, {x:.6g}, {y:.6g}, {z:.6g}" for i, (x, y, z) in sorted(m.noeuds.items())]
    lignes.append("*ELEMENT, TYPE=S8R, ELSET=EALL")
    lignes += [f"{i}, " + ", ".join(str(n) for n in conn) for i, conn in sorted(m.elements.items())]
    lignes += ["*NSET, NSET=ENCASTRE"] + _par_lignes(sorted(m.encastres))
    lignes += [
        f"*MATERIAL, NAME={m.materiau.nom}",
        "*ELASTIC",
        f"{m.materiau.E}, {m.materiau.nu}",
        "*DENSITY",
        f"{m.materiau.rho}",
    ]
    groupes: dict[float, list[int]] = {}
    for e in sorted(m.elements):
        groupes.setdefault(round((m.epaisseurs or {}).get(e, m.epaisseur), 4), []).append(e)
    for k, (t, ids) in enumerate(sorted(groupes.items())):
        lignes += [f"*ELSET, ELSET=EP{k}"] + _par_lignes(ids)
        lignes += [f"*SHELL SECTION, ELSET=EP{k}, MATERIAL={m.materiau.nom}", f"{t}"]
    lignes += [
        "*BOUNDARY",
        "ENCASTRE, 1, 6",
    ]
    if m.appuis_x:
        lignes[lignes.index("*BOUNDARY"):lignes.index("*BOUNDARY")] = ["*NSET, NSET=APPUI"] + _par_lignes(sorted(m.appuis_x))
        lignes.append("APPUI, 1, 1")
    if analyse == "statique":
        lignes.append("*STEP")
        lignes.append("*STATIC")
    elif analyse == "flambement":
        lignes.append("*STEP")
        lignes.append("*BUCKLE")
        lignes.append(f"{n_modes}")
    else:
        raise ValueError(analyse)
    lignes.append("*CLOAD")
    for n, f in sorted(m.forces.items()):
        for ddl, val in enumerate(f, start=1):
            if val != 0.0:
                lignes.append(f"{n}, {ddl}, {val:.9g}")
    if fichiers:
        lignes += ["*NODE FILE", "U", "*EL FILE", "S"]
    if analyse == "statique":
        lignes += ["*NODE PRINT, NSET=NALL", "U", "*NODE PRINT, NSET=ENCASTRE", "RF"]
        if m.appuis_x:
            lignes += ["*NODE PRINT, NSET=APPUI", "RF"]
    lignes.append("*END STEP")
    chemin.write_text("\n".join(lignes) + "\n")


def _par_lignes(ids: list[int], n: int = 12) -> list[str]:
    return [", ".join(str(i) for i in ids[k : k + n]) for k in range(0, len(ids), n)]


def lancer(modele: ModeleCoque, dossier: Path, nom: str = "job", analyse: str = "statique", n_modes: int = 3,
           fichiers: bool = True):
    """Lance CalculiX. fichiers=False : pas de .frd (plus rapide), seuls le .dat et les facteurs sont lus."""
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / f"{nom}.frd").unlink(missing_ok=True)
    ecrire_inp(modele, dossier / f"{nom}.inp", analyse, n_modes, fichiers)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("GABARIT_THREADS", str(os.cpu_count() or 1)))
    proc = subprocess.run(
        [trouver_ccx(), "-i", nom], cwd=dossier, capture_output=True, text=True, timeout=3600, env=env
    )
    if proc.returncode != 0 or "*ERROR" in proc.stdout:
        raise RuntimeError(f"CalculiX a échoué :\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
    return Resultats(dossier / f"{nom}.frd", dossier / f"{nom}.dat")


class Resultats:
    """Lecture du .frd (déplacements, contraintes aux noeuds des coques expansées) et du .dat."""

    def __init__(self, frd: Path, dat: Path):
        self.coords: dict[int, np.ndarray] = {}
        self.blocs: list[tuple[str, dict[int, np.ndarray]]] = []
        if frd.exists():
            self._lire_frd(frd)
        self.facteurs_flambement = _lire_facteurs(dat)
        self.u_noeuds, self.reactions = _lire_dat_statique(dat)

    def _lire_frd(self, chemin: Path) -> None:
        lignes = chemin.read_text().splitlines()
        i = 0
        while i < len(lignes):
            ligne = lignes[i]
            if ligne.startswith("    2C"):
                i += 1
                while not lignes[i].startswith(" -3"):
                    l = lignes[i]
                    self.coords[int(l[3:13])] = np.array([float(l[13:25]), float(l[25:37]), float(l[37:49])])
                    i += 1
            elif ligne.startswith(" -4"):
                nom = ligne[5:13].strip()
                i += 1
                while lignes[i].startswith(" -5"):
                    i += 1
                valeurs: dict[int, np.ndarray] = {}
                while not lignes[i].startswith(" -3"):
                    l = lignes[i]
                    if l.startswith(" -1"):
                        n = int(l[3:13])
                        champs = [float(l[13 + 12 * k : 25 + 12 * k]) for k in range((len(l) - 13) // 12)]
                        valeurs[n] = np.array(champs)
                    i += 1
                self.blocs.append((nom, valeurs))
            i += 1

    def champ(self, nom: str, indice: int = 0) -> dict[int, np.ndarray]:
        blocs = [v for n, v in self.blocs if n == nom]
        return blocs[indice]

    def deplacements(self, indice: int = 0) -> dict[int, np.ndarray]:
        return self.champ("DISP", indice)

    def von_mises(self) -> dict[int, float]:
        res = {}
        for n, s in self.champ("STRESS").items():
            sxx, syy, szz, sxy, syz, szx = s[:6]
            res[n] = float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                                   + 3 * (sxy**2 + syz**2 + szx**2)))
        return res


def _lire_facteurs(dat: Path) -> list[float]:
    if not dat.exists():
        return []
    texte = dat.read_text()
    if "B U C K L I N G" not in texte:
        return []
    bloc = texte.split("B U C K L I N G")[1]
    return [float(m.group(2)) for m in re.finditer(r"^\s+(\d+)\s+([-\d.Ee+]+)\s*$", bloc, re.M)]


def _lire_dat_statique(dat: Path) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """Déplacements des noeuds d'origine (coques non expansées) et réactions par jeu de noeuds."""
    u: dict[int, np.ndarray] = {}
    rf: dict[str, dict[int, np.ndarray]] = {}
    if not dat.exists():
        return u, rf
    cible = None
    for ligne in dat.read_text().splitlines():
        if "displacements (vx,vy,vz) for set" in ligne:
            cible = u
            continue
        mo = re.search(r"forces \(fx,fy,fz\) for set (\S+)", ligne)
        if mo:
            cible = rf.setdefault(mo.group(1), {})
            continue
        champs = ligne.split()
        if cible is not None and len(champs) == 4 and champs[0].isdigit():
            cible[int(champs[0])] = np.array([float(c) for c in champs[1:]])
        elif champs:
            cible = None
    return u, rf
