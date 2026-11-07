"""Schéma du cahier des charges structuré (phase 0).

Unités : mm, N, MPa, kg. Repère du montage : X horizontal, sortant de la face d'encastrement ;
Y horizontal, le long de la face ; Z vertical vers le haut. Chaque valeur non fournie par le sujet
est marquée comme hypothèse dans `hypotheses`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Materiau(BaseModel):
    nom: str
    E: float = Field(description="Module d'Young, MPa")
    nu: float
    Re: float = Field(description="Limite élastique, MPa")
    rho: float = Field(description="Masse volumique, kg/m3")


class Trou(BaseModel):
    y: float
    z: float


class GrilleFixation(BaseModel):
    face: str = Field(description="Plan de la face d'appui, ex. 'x = 20'")
    taraudage: str
    diametre_mineur: float
    trous: list[Trou]
    utilisation: str


class Encombrement(BaseModel):
    description: str
    y_min: float
    y_max: float
    z_min: float
    x_max: float


class Chargement(BaseModel):
    nom: str
    force: float = Field(description="N, norme")
    direction: tuple[float, float, float]
    point_application: tuple[float, float, float] | None = None
    distance_encastrement: float
    interface: str


class Procede(BaseModel):
    nombre_pieces_max: int
    assemblage: list[Literal["soudure", "boulonnage"]]
    epaisseurs_disponibles: list[float]
    decoupe: str
    pliage: str
    rayon_pliage_interieur: str
    longueur_pliage_max: float | None = None
    ves_disponibles: list[float] = []


class Criteres(BaseModel):
    contrainte: str
    objectif: str
    coefficient_flambement_min: float
    autres: list[str] = []


class CahierDesCharges(BaseModel):
    nom: str
    source: str
    materiau: Materiau
    grille_fixation: GrilleFixation
    encombrement: Encombrement
    chargements: list[Chargement]
    procede: Procede
    criteres: Criteres
    hypotheses: list[str]
    questions_ouvertes: list[str] = []

    @classmethod
    def charger(cls, chemin: str | Path) -> "CahierDesCharges":
        return cls.model_validate(json.loads(Path(chemin).read_text(encoding="utf-8")))
