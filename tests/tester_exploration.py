"""Test hors ligne de l'exploration longue (modèle ouvert), sans réseau.

Un faux serveur local joue un petit modèle sur trois essais : le premier rend un concept valide (enregistré puis
bilan), le deuxième conclut sans concept, le troisième tombe sur une erreur du serveur. On vérifie le catalogue,
que chaque essai repart d'une conversation neuve sur une piste différente, et que la liste de ce qui a déjà été
essayé est transmise au modèle.
Lancer :  python tests/tester_exploration.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

from gabarit.ia.exploration import INVENTER, PISTES, choisir_piste, explorer  # noqa: E402

concept = json.loads((RACINE / "concepts" / "sae51_L_V.json").read_text(encoding="utf-8"))
concept["nom"] = "Joues dégressives test"


def message(contenu: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": contenu}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


REPONSES = [message(f"Voici :\n```json\n{json.dumps(concept)}\n```"),  # essai 1 : concept valide
            message("Bilan : un concept enregistré."),
            message("Je n'ai pas trouvé de forme fabricable pour cette piste."),  # essai 2 : pas de concept
            None]  # essai 3 : erreur du serveur
requetes: list[dict] = []


class Serveur(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        requetes.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
        r = REPONSES[len(requetes) - 1]
        code, data = (500, b'{"error": "modele plante"}') if r is None else (200, json.dumps(r).encode())
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), Serveur)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        cat = explorer(heures=1, essais_max=3, url=f"http://127.0.0.1:{srv.server_port}/v1", modele="faux",
                       sortie=Path(tmp) / "concepts", dossier=Path(tmp) / "exploration", outils=False,
                       calcul=False, afficher=lambda *a, **k: None)
        with cat.open(encoding="utf-8-sig") as f:
            lignes = list(csv.DictReader(f))
        enregistres = sorted(p.name for p in (Path(tmp) / "concepts").glob("*.json"))
        journaux = sorted(p.parent.name for p in (Path(tmp) / "exploration").glob("essai_*/journal.md"))
    demande_2 = requetes[2]["messages"][1]["content"]
    faux_catalogue = [{"piste": p, "statut": "enregistré" if k == 0 else "aucun concept"} for k, p in enumerate(PISTES)]
    controles = {
        "3 essais au catalogue": [ligne["statut"] for ligne in lignes] == ["enregistré", "aucun concept", "erreur"],
        "une piste différente par essai": [ligne["piste"] for ligne in lignes] == PISTES[:3],
        "concept enregistré": enregistres == ["joues_degressives_test.json"],
        "masse mesurée": float(lignes[0]["masse_kg"]) > 0,
        "conversation neuve à chaque essai": len(requetes[2]["messages"]) == 2,
        "pistes déjà essayées transmises": "Déjà essayé" in demande_2 and "joues dégressives : enregistré" in demande_2,
        "un journal par essai": journaux == ["essai_001", "essai_002", "essai_003"],
        "piste inventée après la liste": choisir_piste(len(PISTES) + 1, faux_catalogue) == INVENTER,
        "puis reprise d'une piste ratée": choisir_piste(len(PISTES) + 2, faux_catalogue) == PISTES[1],
    }
    for nom, ok in controles.items():
        print(f"{'OK ' if ok else 'ÉCHEC'} {nom}")
    return 0 if all(controles.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
