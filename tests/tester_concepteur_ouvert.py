"""Test hors ligne de l'IA Concepteur avec un modèle ouvert (API compatible OpenAI), sans réseau.

Un faux serveur local joue un modèle ouvert dans deux scénarios :
1. avec appels d'outils : un concept cassé à vérifier, puis le concept corrigé à enregistrer, puis un bilan ;
2. sans outils (le serveur les refuse pour ce modèle) : le concept arrive dans un bloc ```json du texte,
   d'abord cassé (le rapport repart au modèle), puis corrigé (il est enregistré), puis un bilan.
Lancer :  python tests/tester_concepteur_ouvert.py
"""
from __future__ import annotations

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

from gabarit.ia.ouvert import concevoir_ouvert  # noqa: E402

concept = json.loads((RACINE / "concepts" / "sae51_L_V.json").read_text(encoding="utf-8"))
concept["nom"] = "Test modèle ouvert"
casse = json.loads(json.dumps(concept))
del casse["fixation"]


def message(contenu="", appels=None):
    m = {"role": "assistant", "content": contenu}
    if appels:
        m["tool_calls"] = [{"id": f"appel_{i}", "type": "function",
                            "function": {"name": nom, "arguments": json.dumps({"concept": c})}}
                           for i, (nom, c) in enumerate(appels)]
    return {"choices": [{"message": m, "finish_reason": "tool_calls" if appels else "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


SCENARIOS = {
    "outils": [message("", [("verifier_concept", casse)]),
               message("Corrigé.", [("enregistrer_concept", concept)]),
               message("Bilan : un concept enregistré.")],
    "texte": [None,  # première requête avec outils : refusée par le serveur
              message(f"<think>je réfléchis</think>Voici :\n```json\n{json.dumps(casse)}\n```"),
              message(f"Corrigé :\n```json\n{json.dumps(concept)}\n```"),
              message("Bilan : un concept enregistré.")],
}
requetes: list[dict] = []
scenario = {"nom": "outils"}


class Serveur(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        corps = json.loads(self.rfile.read(int(self.headers["content-length"])))
        requetes.append(corps)
        reponse = SCENARIOS[scenario["nom"]][len(requetes) - 1]
        if reponse is None:
            code, data = 400, json.dumps({"error": "model does not support tools"}).encode()
        else:
            code, data = 200, json.dumps(reponse).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def jouer(nom: str) -> tuple:
    scenario["nom"] = nom
    requetes.clear()
    with tempfile.TemporaryDirectory() as tmp:
        s = concevoir_ouvert(1, url=URL, modele="faux-modele", sortie=Path(tmp) / "concepts",
                             dossier_journal=Path(tmp) / "journal", calcul=False, afficher=lambda *a, **k: None)
        noms = [json.loads(p.read_text(encoding="utf-8"))["nom"] for p in s.concepts]
    return s, noms, list(requetes)


srv = HTTPServer(("127.0.0.1", 0), Serveur)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_port}/v1"


def main() -> int:
    s1, noms1, req1 = jouer("outils")
    s2, noms2, req2 = jouer("texte")
    retour_outil = req1[1]["messages"][-1]
    controles = {
        "outils : 3 requêtes": len(req1) == 3,
        "outils : outils déclarés": [t["function"]["name"] for t in req1[0]["tools"]]
                                    == ["verifier_concept", "enregistrer_concept", "predimensionner"],
        "outils : erreur renvoyée au modèle": retour_outil["role"] == "tool" and "fixation" in retour_outil["content"],
        "outils : concept enregistré": noms1 == ["Test modèle ouvert"],
        "outils : arrêt normal": s1.arret == "terminé" and s1.bilan.startswith("Bilan"),
        "texte : bascule sans outils": "tools" not in req2[1],
        "texte : rapport renvoyé au modèle": "refusé" in req2[2]["messages"][-1]["content"],
        "texte : concept enregistré": noms2 == ["Test modèle ouvert"],
        "texte : arrêt normal": s2.arret == "terminé" and s2.bilan.startswith("Bilan"),
        "jetons comptés": s1.jetons["entree"] == 3000,
    }
    for nom, ok in controles.items():
        print(f"{'OK ' if ok else 'ÉCHEC'} {nom}")
    return 0 if all(controles.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
