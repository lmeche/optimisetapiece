"""Interface minimale vers CalculiX : écriture du .inp, lancement de ccx, lecture des résultats.

Le modèle est décrit en coques (S8R, 8 noeuds) : c'est la représentation retenue pour les tôles
de l'ensemble mécano-soudé. Le format est volontairement simple pour être généré par programme.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import warnings
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
               fichiers: bool = True, details: bool = False) -> None:
    """details=True : en statique, contraintes aux points d'intégration et énergie de chaque élément (.dat) ;
    en flambement, déformée de chaque mode aux noeuds d'origine (.dat)."""
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
        if details:
            lignes += ["*EL PRINT, ELSET=EALL", "S"]
        lignes += ["*EL PRINT, ELSET=EALL", "ELSE"]  # toujours : sert au bilan d'énergie de `lancer`
    elif details:
        lignes += ["*NODE PRINT, NSET=NALL", "U"]
    lignes.append("*END STEP")
    chemin.write_text("\n".join(lignes) + "\n")


def _par_lignes(ids: list[int], n: int = 12) -> list[str]:
    return [", ".join(str(i) for i in ids[k : k + n]) for k in range(0, len(ids), n)]


TOLERANCE_ENERGIE = 0.05  # |énergie des éléments / travail des efforts - 1| toléré (Clapeyron)
TOLERANCE_EQUILIBRE = 1e-3  # |efforts + réactions| / |efforts| toléré (équilibre global en translation)
ESSAIS_CALCUL = 3  # calculs statiques lancés au plus avant de conclure à une solution fausse


def bilan_statique(modele: ModeleCoque, res: Resultats) -> dict[str, float]:
    """Contrôles d'une solution statique linéaire, les déplacements imposés étant nuls :
    - energie : somme des énergies des éléments / (1/2 somme f.u) - 1, nul par le théorème de Clapeyron ;
    - equilibre : |somme des efforts + somme des réactions| / |somme des efforts|, nul quand K u = f est
      résolu exactement (les efforts intérieurs d'une structure s'équilibrent pour tout champ u).
    Une solution juste donne 1e-7 aux deux. Le solveur SPOOLES multi-tâche (CalculiX 2.21) rend parfois, pour un
    même fichier, une solution fausse : 3 calculs sur 40 du modèle L + V avec 4 tâches (énergie +2 %, +6 %,
    +1 563 % ; équilibre faux de 4 à 18 % de l'effort), aucun sur 40 avec une tâche. Le contrôle d'équilibre est le
    plus sûr des deux : il a pris les trois, celui d'énergie seul en aurait laissé passer un."""
    travail = 0.5 * sum(float(np.dot(f, res.u_noeuds[n])) for n, f in modele.forces.items())
    energie = float(sum(res.energie.values()))
    # Équilibre de chaque ddl : effort intérieur = effort extérieur (+ réaction s'il est bloqué). CalculiX imprime en
    # RF l'effort intérieur des noeuds du jeu, sur leurs 3 ddl (vérifié : un effort posé sur un noeud encastré n'y
    # figure pas). Les efforts intérieurs s'équilibrant, efforts sur les ddl libres + RF des ddl bloqués = 0.
    x_seul = np.array([1.0, 0.0, 0.0])
    efforts, reactions = np.zeros(3), np.zeros(3)
    for n, f in modele.forces.items():
        if n not in modele.encastres:
            efforts += np.asarray(f, dtype=float) * ((1.0 - x_seul) if n in modele.appuis_x else 1.0)
    for n, rf in res.reactions.get("ENCASTRE", {}).items():
        reactions += rf
    for n, rf in res.reactions.get("APPUI", {}).items():
        if n not in modele.encastres:
            reactions += rf * x_seul
    return {
        "energie": energie / travail - 1 if travail > 0 else math.inf,
        "equilibre": float(np.linalg.norm(efforts + reactions) / max(np.linalg.norm(efforts), 1e-30)),
        "energie_Nmm": energie,
        "travail_Nmm": travail,
    }


def bilan_correct(bilan: dict[str, float]) -> bool:
    return abs(bilan["energie"]) <= TOLERANCE_ENERGIE and bilan["equilibre"] <= TOLERANCE_EQUILIBRE


def lancer(modele: ModeleCoque, dossier: Path, nom: str = "job", analyse: str = "statique", n_modes: int = 3,
           fichiers: bool = True, details: bool = False):
    """Lance CalculiX. fichiers=False : pas de .frd (plus rapide), seuls le .dat et les facteurs sont lus.
    details=True : voir `ecrire_inp` (lu dans Resultats.contraintes_ip, .energie, .modes).
    Un calcul statique chargé est contrôlé (`bilan_statique`, rangé dans Resultats.bilan) et relancé tant que le
    bilan est faux, jusqu'à ESSAIS_CALCUL fois ; au-delà (mécanisme, matrice singulière...) : RuntimeError."""
    dossier.mkdir(parents=True, exist_ok=True)
    ecrire_inp(modele, dossier / f"{nom}.inp", analyse, n_modes, fichiers, details)
    # une seule tâche par défaut : SPOOLES multi-tâche se trompe parfois (voir `bilan_statique`) et gagne peu sur
    # ces modèles (L + V, 7 000 noeuds : 1,3 s en une tâche, 1,1 s en deux, 1,1 à 1,3 s en quatre) ;
    # GABARIT_THREADS pour changer
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("GABARIT_THREADS", "1"))
    controle = analyse == "statique" and bool(modele.forces)
    bilans = []
    for essai in range(1, (ESSAIS_CALCUL if controle else 1) + 1):
        for ext in ("frd", "dat"):  # jamais de résultats d'un calcul précédent
            (dossier / f"{nom}.{ext}").unlink(missing_ok=True)
        proc = subprocess.run(
            [trouver_ccx(), "-i", nom], cwd=dossier, capture_output=True, text=True, timeout=3600, env=env
        )
        if proc.returncode != 0 or "*ERROR" in proc.stdout:
            erreurs = [ligne.strip() for ligne in proc.stdout.splitlines() if "ERROR" in ligne]
            raise RuntimeError(f"CalculiX a échoué ({'; '.join(erreurs[:3]) or f'code {proc.returncode}'}) :\n"
                               f"{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
        res = Resultats(dossier / f"{nom}.frd", dossier / f"{nom}.dat")
        res.essais = essai
        if not controle:
            return res
        res.bilan = bilan_statique(modele, res)
        if bilan_correct(res.bilan):
            return res
        bilans.append(res.bilan)
        if essai < ESSAIS_CALCUL:
            warnings.warn(f"CalculiX ({dossier / nom}) : solution statique fausse (énergie {res.bilan['energie']:+.2%}, "
                          f"équilibre {res.bilan['equilibre']:.1e}), calcul relancé", RuntimeWarning, stacklevel=2)
    raise RuntimeError(f"CalculiX : solution statique fausse à chacun des {ESSAIS_CALCUL} essais (écarts d'énergie "
                       + ", ".join(f"{b['energie']:+.2%}" for b in bilans) + ") : mécanisme ou matrice singulière ?")


class Resultats:
    """Lecture du .frd (déplacements, contraintes aux noeuds des coques expansées) et du .dat."""

    def __init__(self, frd: Path, dat: Path):
        self.coords: dict[int, np.ndarray] = {}
        self.blocs: list[tuple[str, dict[int, np.ndarray]]] = []
        if frd.exists():
            self._lire_frd(frd)
        self.facteurs_flambement = _lire_facteurs(dat)
        self.u_noeuds, self.reactions = _lire_dat_statique(dat)
        self.contraintes_ip, self.energie, self.modes = _lire_dat_details(dat)
        self.bilan: dict[str, float] | None = None  # contrôles du calcul statique (voir `bilan_statique`)
        self.essais = 1  # calculs lancés pour obtenir ce résultat

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


def _lire_dat_details(dat: Path) -> tuple[dict[int, np.ndarray], dict[int, float], list[dict[int, np.ndarray]]]:
    """Sorties détaillées du .dat (vides si absentes) :
    - contraintes aux 8 points d'intégration de chaque élément (coque expansée en brique C3D20R), tableau
      (8, 6) des composantes sxx, syy, szz, sxy, sxz, syz dans le repère global. Ordre des points (vérifié sur
      une console en flexion) : le premier indice varie le plus vite, les points 1 à 4 sont sur la demi-épaisseur
      côté -normale (zeta = -1/racine(3)), les points 5 à 8 côté +normale ;
    - énergie de déformation de chaque élément (N.mm) ;
    - déformée de chaque mode de flambement aux noeuds d'origine, dans l'ordre des modes."""
    contraintes: dict[int, np.ndarray] = {}
    energie: dict[int, float] = {}
    modes: list[dict[int, np.ndarray]] = []
    if not dat.exists():
        return contraintes, energie, modes
    texte = dat.read_text()
    if "integ.pnt." not in texte and "internal energy" not in texte and "E I G E N V A L U E" not in texte:
        return contraintes, energie, modes
    cible = None
    brut: dict[int, list[list[float]]] = {}
    for ligne in texte.splitlines():
        if "stresses (elem, integ.pnt." in ligne:
            cible = "S"
            continue
        if "internal energy (element, energy)" in ligne:
            cible = "E"
            continue
        if "E I G E N V A L U E" in ligne:
            cible = "attente_mode"
            continue
        if "displacements (vx,vy,vz)" in ligne:
            if cible == "attente_mode":
                modes.append({})
                cible = "U"
            else:
                cible = None
            continue
        champs = ligne.split()
        if not champs:  # lignes vides autour des blocs
            continue
        if cible == "S" and len(champs) >= 8 and champs[0].isdigit():
            brut.setdefault(int(champs[0]), []).append([float(c) for c in champs[2:8]])
        elif cible == "E" and len(champs) == 2 and champs[0].isdigit():
            energie[int(champs[0])] = float(champs[1])
        elif cible == "U" and len(champs) == 4 and champs[0].isdigit():
            modes[-1][int(champs[0])] = np.array([float(c) for c in champs[1:]])
        elif cible in ("S", "E", "U"):
            cible = None
    for e, pts in brut.items():
        contraintes[e] = np.array(pts)
    return contraintes, energie, modes
