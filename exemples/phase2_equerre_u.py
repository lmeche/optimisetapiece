"""Phase 2 : évaluation de l'équerre en U de référence sur la SAE 5.1.

Lancer depuis la racine du dépôt :  python exemples/phase2_equerre_u.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gabarit.cahier_des_charges import CahierDesCharges  # noqa: E402
from gabarit.concepts.equerre_u import EquerreU  # noqa: E402
from gabarit.evaluation import evaluer  # noqa: E402
from gabarit.fem.calculix import Resultats  # noqa: E402
from gabarit.rapport.image import image_von_mises  # noqa: E402

racine = Path(__file__).resolve().parents[1]
cdc = CahierDesCharges.charger(racine / "cahiers" / "sae51_equerre.json")
concept = EquerreU()
sortie = racine / "resultats" / "phase2"
t0 = time.time()
r = evaluer(concept, cdc, taille=6.0, dossier=sortie / "calcul")
print(f"Calcul en {time.time() - t0:.0f} s")
print(json.dumps({"parametres": concept.parametres(), "resultat": asdict(r)}, indent=2, ensure_ascii=False, default=float))
res = Resultats(sortie / "calcul" / "statique.frd", sortie / "calcul" / "statique.dat")
img = image_von_mises(res, sortie / "equerre_u_von_mises.png", "Équerre en U de référence, 9 810 N")
print("Image :", img)
