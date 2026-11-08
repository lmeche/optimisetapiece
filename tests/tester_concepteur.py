"""Phase 4 : test hors ligne de la boucle de l'IA Concepteur (sans clé API, sans réseau).

Un faux serveur local joue le rôle de l'API : il répond en flux SSE par un scénario fixé (un concept
cassé à vérifier, une réponse coupée par max_tokens, le concept corrigé à vérifier, un relais vers le modèle de secours puis l'enregistrement,
un bilan). On vérifie que le vrai SDK anthropic envoie bien les paramètres prévus, que les résultats
d'outils reviennent à l'IA, que le bloc de relais est filtré et que le concept est enregistré.
Lancer :  python tests/tester_concepteur.py
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
import anthropic  # noqa: E402

from gabarit.ia.concepteur import concevoir  # noqa: E402

concept = json.loads((RACINE / "concepts" / "sae51_L_V.json").read_text(encoding="utf-8"))
concept["nom"] = "Test L + V"
casse = json.loads(json.dumps(concept))
del casse["appui"]


def sse(blocs, stop):
    ev = [("message_start", {"type": "message_start", "message": {"id": "msg_x", "type": "message", "role": "assistant",
          "model": "claude-opus-5-5", "content": [], "stop_reason": None, "stop_sequence": None,
          "usage": {"input_tokens": 100, "output_tokens": 1, "cache_read_input_tokens": 80}}})]
    for i, b in enumerate(blocs):
        if b["type"] == "text":
            ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}})]
        elif b["type"] == "thinking":
            ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": b["thinking"]}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": "sig"}})]
        elif b["type"] == "tool_use":
            ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})]
        else:
            ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": b})]
        ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    ev += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 50}}),
           ("message_stop", {"type": "message_stop"})]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in ev).encode()

TOURS = [
    ([{"type": "thinking", "thinking": "Je pars du L + V."}, {"type": "text", "text": "Je vérifie un premier jet."},
      {"type": "tool_use", "id": "toolu_1", "name": "verifier_concept", "input": {"concept": casse}}], "tool_use"),
    ([{"type": "text", "text": "Je rédige tout d'un coup."},  # réponse coupée : à relancer, pas à exécuter
      {"type": "tool_use", "id": "toolu_x", "name": "enregistrer_concept", "input": {"concept": {"nom": "coupé"}}}],
     "max_tokens"),
    ([{"type": "tool_use", "id": "toolu_2", "name": "verifier_concept", "input": {"concept": concept}}], "tool_use"),
    ([{"type": "text", "text": "partiel"}, {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-4-8"}},
      {"type": "tool_use", "id": "toolu_3", "name": "enregistrer_concept", "input": {"concept": concept}}], "tool_use"),
    ([{"type": "text", "text": "Bilan : un concept L + V enregistré."}], "end_turn"),
]
requetes = []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        corps = json.loads(self.rfile.read(int(self.headers["content-length"])))
        requetes.append((dict(self.headers), corps))
        blocs, stop = TOURS[len(requetes) - 1]
        data = sse(blocs, stop)
        self.send_response(200); self.send_header("content-type", "text/event-stream"); self.send_header("content-length", str(len(data))); self.end_headers()
        self.wfile.write(data)


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{srv.server_port}", max_retries=0)
    with tempfile.TemporaryDirectory() as tmp:
        s = concevoir(1, sortie=Path(tmp) / "concepts", dossier_journal=Path(tmp) / "journal", client=client,
                      calcul=False, afficher=lambda *a, **k: None)
        enregistres = [json.loads(p.read_text(encoding="utf-8"))["nom"] for p in s.concepts]
    _, premiere = requetes[0]
    entetes = {k.lower(): v for k, v in requetes[0][0].items()}
    retour_1 = requetes[1][1]["messages"][-1]["content"][0]
    renvoi_3 = [b["type"] for b in requetes[4][1]["messages"][-2]["content"]]
    coupee, relance = requetes[1][1], requetes[2][1]
    controles = {
        "5 requêtes": len(requetes) == 5,
        "réponse coupée relancée plus longue": relance["max_tokens"] > coupee["max_tokens"]
                                              and relance["messages"] == coupee["messages"],
        "arrêt normal": s.arret == "terminé",
        "concept enregistré": enregistres == ["Test L + V"],
        "erreur de structure renvoyée à l'IA": "appui" in retour_1["content"],
        "modèle": premiere["model"] == "claude-opus-5-5",
        "réflexion adaptative": premiere["thinking"]["type"] == "adaptive",
        "effort explicite": "effort" in premiere["output_config"],
        "relais de secours": premiere["fallbacks"] == "default"
                              and "server-side-fallback-2026-07-01" in entetes.get("anthropic-beta", ""),
        "cache du prompt": "cache_control" in premiere["system"][0],
        "bloc de relais filtré": renvoi_3 == ["text", "tool_use"],
        "bilan lu": s.bilan.startswith("Bilan"),
    }
    for nom, ok in controles.items():
        print(f"{'OK ' if ok else 'ÉCHEC'} {nom}")
    return 0 if all(controles.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
