"""Croquis des résultats : nuage des noeuds coloré par la contrainte de von Mises, déformée amplifiée."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gabarit.fem.calculix import Resultats  # noqa: E402


def image_von_mises(res: Resultats, chemin: Path, titre: str, re: float = 235.0, amplification: float = 10.0,
                    vues=((25, -60), (90, -90))) -> Path:
    vm = res.von_mises()
    U = res.deplacements()
    ids = list(vm)
    p = np.array([res.coords[n] + amplification * U[n] for n in ids])
    v = np.array([vm[n] for n in ids])
    fig = plt.figure(figsize=(12, 5.2))
    for k, (elev, azim) in enumerate(vues):
        ax = fig.add_subplot(1, len(vues), k + 1, projection="3d")
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=np.minimum(v, 1.5 * re), cmap="turbo", s=1.5,
                        vmin=0, vmax=1.5 * re)
        ax.set_box_aspect(np.ptp(p, axis=0))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    fig.colorbar(sc, ax=fig.axes, shrink=0.7, label=f"von Mises (MPa), plafonné à 1,5 x Re = {1.5 * re:.0f}")
    fig.suptitle(f"{titre} (déformée x {amplification:g})")
    fig.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return chemin
