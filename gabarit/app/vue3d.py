"""Figures Plotly de l'application : maquette 3D d'un concept, grille de fixation, nuage des essais."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

COULEURS = ["#2a6fdb", "#e8833a", "#3aa76d", "#a05cc8"]


def figure_faces(faces: list[dict], x_face: float = 20.0, x_charge: float | None = 320.0,
                 hauteur: int = 460) -> go.Figure:
    """faces : [{"id": "L.dos", "tole": "L", "t": 4.0, "points": [[x, y, z], ...]}, ...]"""
    fig = go.Figure()
    toles = sorted({f["tole"] for f in faces})
    deja = set()
    pts_tous = np.vstack([np.asarray(f["points"], float) for f in faces]) if faces else np.zeros((1, 3))
    for f in faces:
        p = np.asarray(f["points"], float)
        n = len(p)
        coul = COULEURS[toles.index(f["tole"]) % len(COULEURS)]
        nom = f"Tôle {f['tole']} ({f['t']:g} mm)"
        fig.add_trace(go.Mesh3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], i=[0] * (n - 2), j=list(range(1, n - 1)),
                                k=list(range(2, n)), color=coul, opacity=0.85, flatshading=True, name=nom,
                                legendgroup=f["tole"], showlegend=f["tole"] not in deja,
                                hovertext=f"{f['id']} : {f['t']:g} mm", hoverinfo="text"))
        deja.add(f["tole"])
        bord = np.vstack([p, p[:1]])
        fig.add_trace(go.Scatter3d(x=bord[:, 0], y=bord[:, 1], z=bord[:, 2], mode="lines",
                                   line={"color": "#1d1d1f", "width": 2}, hoverinfo="skip", showlegend=False))
    # face du socle et appui
    zmin, zmax = 0.0, max(200.0, float(pts_tous[:, 2].max()) + 10)
    fig.add_trace(go.Mesh3d(x=[x_face] * 4, y=[-80, 80, 80, -80], z=[zmin, zmin, zmax, zmax], i=[0, 0], j=[1, 2],
                            k=[2, 3], color="#9aa0a6", opacity=0.25, name="Face du socle", hoverinfo="name"))
    if x_charge is not None:
        zc = float(pts_tous[:, 2].max())
        a = np.linspace(0, 2 * math.pi, 40)
        fig.add_trace(go.Scatter3d(x=x_charge + 25 * np.cos(a), y=25 * np.sin(a), z=np.full_like(a, zc + 1),
                                   mode="lines", line={"color": "#d93025", "width": 5}, name="Appui Ø50 (1 t)"))
        fig.add_trace(go.Cone(x=[x_charge], y=[0], z=[zc + 45], u=[0], v=[0], w=[-40], sizemode="absolute",
                              sizeref=18, anchor="tip", colorscale=[[0, "#d93025"], [1, "#d93025"]],
                              showscale=False, hoverinfo="skip", showlegend=False))
    fig.update_layout(height=hauteur, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                      legend={"orientation": "h", "y": 1.02, "x": 0},
                      scene={"aspectmode": "data", "xaxis_title": "X (mm)", "yaxis_title": "Y (mm)",
                             "zaxis_title": "Z (mm)", "camera": {"eye": {"x": 1.4, "y": -1.6, "z": 0.9}}})
    return fig


def faces_du_concept(concept) -> list[dict]:
    return [{"id": f.id, "tole": f.tole, "t": f.t, "points": f.points.tolist()} for f in concept.geometrie().faces]


def faces_de_synthese(geometrie: dict) -> list[dict]:
    return [{"id": k, "tole": k.split(".")[0], "t": v["epaisseur"], "points": v["points"]} for k, v in geometrie.items()]


def figure_grille(cdc) -> go.Figure:
    ys = [t.y for t in cdc.grille_fixation.trous]
    zs = [t.z for t in cdc.grille_fixation.trous]
    fig = go.Figure(go.Scatter(x=ys, y=zs, mode="markers", marker={"size": 11, "color": "#2a6fdb"},
                               hovertemplate="y = %{x}, z = %{y}<extra></extra>"))
    e = cdc.encombrement
    fig.add_shape(type="rect", x0=e.y_min, x1=e.y_max, y0=e.z_min, y1=max(zs) + 40,
                  line={"color": "#9aa0a6", "dash": "dot"})
    fig.update_layout(height=300, margin={"l": 40, "r": 10, "t": 10, "b": 40}, xaxis_title="Y (mm)",
                      yaxis_title="Z (mm)", xaxis={"range": [e.y_min - 30, e.y_max + 30]},
                      yaxis={"scaleanchor": "x"})
    return fig


def lire_essais(chemin: Path) -> list[dict]:
    if not chemin.exists():
        return []
    with chemin.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def figure_essais(essais: list[dict], hauteur: int = 380) -> go.Figure:
    def val(e, k):
        try:
            x = float(e[k])
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    fig = go.Figure()
    for ok, coul, nom in (("False", "#b8bcc2", "non faisable"), ("True", "#2a6fdb", "faisable")):
        pts = [(val(e, "effort_tenu_N"), val(e, "masse_kg"), e["numero"]) for e in essais if e["faisable"] == ok]
        pts = [p for p in pts if p[0] is not None and p[1] is not None]
        fig.add_trace(go.Scatter(x=[p[0] / 1000 for p in pts], y=[p[1] for p in pts], mode="markers", name=nom,
                                 marker={"color": coul, "size": 8}, text=[f"essai {p[2]}" for p in pts],
                                 hovertemplate="%{text}<br>%{x:.1f} kN, %{y:.2f} kg<extra></extra>"))
    fig.add_vline(x=9.81, line_dash="dash", line_color="#1d1d1f", annotation_text="1 t")
    fig.update_layout(height=hauteur, margin={"l": 50, "r": 10, "t": 10, "b": 45},
                      xaxis_title="Effort tenu avant plastification (kN)", yaxis_title="Masse (kg)",
                      legend={"orientation": "h", "y": 1.08})
    return fig
