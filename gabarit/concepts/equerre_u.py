"""Concept de référence « équerre en U » pour la SAE 5.1 (phase 2, écrit à la main).

Une tôle pliée : un dos vissé sur la grille du socle, un dessus horizontal qui reçoit l'adaptateur,
et deux ailes latérales pliées vers le bas depuis le dessus, soudées au dos sur leur arête arrière.
Le modèle est décrit par ses surfaces moyennes (coques).

Repère du socle : face d'appui en x = 20, Y le long de la face, Z vertical, dessus de la semelle en z = 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gmsh
import numpy as np

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.fem.calculix import Materiau, ModeleCoque

X_FACE = 20.0
RAYON_APPUI_VIS = 8.0  # rayon de la zone serrée sous la tête de vis M10


@dataclass
class EquerreU:
    t: float = 4.0  # épaisseur de tôle
    b: float = 70.0  # largeur entre surfaces moyennes des ailes
    z_dessus: float = 170.0  # altitude de la surface moyenne du dessus
    z_bas: float = 70.0  # bas du dos
    h_aile_dos: float = 100.0  # hauteur des ailes contre le dos
    h_aile_bout: float = 30.0  # hauteur des ailes en bout
    x_bout: float = 350.0  # bout du dessus (l'appui Ø50 est centré en x = 320)
    colonnes_vis: tuple[float, ...] = (-20.0, 20.0)

    def verifier(self, cdc: CahierDesCharges) -> list[str]:
        """Règles géométriques et de fabrication ; liste vide = concept valide."""
        e = []
        if self.t not in cdc.procede.epaisseurs_disponibles:
            e.append(f"épaisseur {self.t} absente de l'atelier")
        if self.b / 2 + self.t / 2 > cdc.encombrement.y_max:
            e.append("trop large pour passer entre les nervures")
        if self.h_aile_dos > self.z_dessus - self.z_bas:
            e.append("aile plus haute que le dos")
        if self.z_dessus - max(self.h_aile_dos, self.h_aile_bout) < cdc.encombrement.z_min + 10:
            e.append("aile trop proche de la semelle")
        if self.x_bout > cdc.encombrement.x_max:
            e.append("dépasse la semelle")
        aile_min = 0.7 * 8 * self.t
        for nom, h in (("h_aile_bout", self.h_aile_bout), ("h_aile_dos", self.h_aile_dos)):
            if h < aile_min:
                e.append(f"{nom} < aile minimale de pliage ({aile_min:.0f} mm)")
        if not self.trous_utilises(cdc):
            e.append("aucun trou de la grille utilisable")
        return e

    def trous_utilises(self, cdc: CahierDesCharges) -> list[tuple[float, float]]:
        marge = 12.0
        return [
            (h.y, h.z)
            for h in cdc.grille_fixation.trous
            if any(abs(h.y - c) < 1e-6 for c in self.colonnes_vis)
            and abs(h.y) <= self.b / 2 - marge
            and self.z_bas + marge <= h.z <= self.z_dessus - marge
        ]

    def masse(self, rho: float) -> float:
        xd = X_FACE + self.t / 2
        aire = (
            self.b * (self.z_dessus - self.z_bas)
            + self.b * (self.x_bout - xd)
            + 2 * (self.x_bout - xd) * (self.h_aile_dos + self.h_aile_bout) / 2
        )
        return aire * self.t * rho * 1e-9  # kg

    def mailler(self, taille: float = 6.0):
        xd = X_FACE + self.t / 2
        y0, y1, zt = -self.b / 2, self.b / 2, self.z_dessus
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("equerre_u")
        occ = gmsh.model.occ

        def plaque(pts):
            ids = [occ.addPoint(*p) for p in pts]
            lignes = [occ.addLine(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
            return occ.addPlaneSurface([occ.addCurveLoop(lignes)])

        faces = [
            plaque([(xd, y0, self.z_bas), (xd, y1, self.z_bas), (xd, y1, zt), (xd, y0, zt)]),
            plaque([(xd, y0, zt), (self.x_bout, y0, zt), (self.x_bout, y1, zt), (xd, y1, zt)]),
        ]
        for y in (y0, y1):
            faces.append(plaque([(xd, y, zt), (self.x_bout, y, zt),
                                 (self.x_bout, y, zt - self.h_aile_bout), (xd, y, zt - self.h_aile_dos)]))
        occ.fragment([(2, f) for f in faces], [])
        occ.synchronize()
        # la subdivision en quadrangles coupe chaque maille en 4 : on maille à 2 x taille
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2 * taille)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2 * taille)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        noeuds = {int(t): tuple(coords[3 * k : 3 * k + 3]) for k, t in enumerate(tags)}
        etags, en = gmsh.model.mesh.getElementsByType(16)
        elements = {int(e): [int(n) for n in en[8 * k : 8 * k + 8]] for k, e in enumerate(etags)}
        gmsh.finalize()
        utilises = {n for c in elements.values() for n in c}
        return {n: c for n, c in noeuds.items() if n in utilises}, elements

    def modele(self, cdc: CahierDesCharges, taille: float = 6.0) -> tuple[ModeleCoque, dict]:
        noeuds, elements = self.mailler(taille)
        xd = X_FACE + self.t / 2
        mat = cdc.materiau
        materiau = Materiau(mat.nom, mat.E, mat.nu, rho=mat.rho * 1e-12, Re=mat.Re)
        trous = self.trous_utilises(cdc)
        encastres = {
            n for n, (x, y, z) in noeuds.items()
            if abs(x - xd) < 1e-6 and any(np.hypot(y - ty, z - tz) <= RAYON_APPUI_VIS for ty, tz in trous)
        }
        ch = cdc.chargements[0]
        x_charge = X_FACE + ch.distance_encastrement
        disque = [n for n, (x, y, z) in noeuds.items()
                  if abs(z - self.z_dessus) < 1e-6 and np.hypot(x - x_charge, y) <= 25.0]
        f = np.array(ch.direction) * ch.force / len(disque)
        forces = {n: tuple(f) for n in disque}
        dos = {n for n, (x, y, z) in noeuds.items() if abs(x - xd) < 1e-6} - encastres
        info = {"trous": trous, "dos": dos, "x_charge": x_charge, "noeuds_charge": disque,
                "x_dos": xd, "t_dos": self.t, "z_charge": self.z_dessus, "t_charge": self.t}
        return ModeleCoque(noeuds, elements, self.t, materiau, encastres, forces), info

    def parametres(self) -> dict:
        return asdict(self)
