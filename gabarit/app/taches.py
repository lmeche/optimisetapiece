"""Tâches longues (IA Concepteur, optimisation, apprentissage) lancées dans des processus séparés.

Gmsh et CalculiX ne supportent pas les fils d'exécution : chaque tâche est un script Python lancé à
part, dont la sortie est lue ligne à ligne par un fil de lecture et mise en file pour l'interface.
Les tâches vivent au niveau du module : recharger la page ne les perd pas.
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]


class Tache:
    def __init__(self, nom: str):
        self.nom = nom
        self.proc: subprocess.Popen | None = None
        self.lignes: list[str] = []
        self.file: queue.Queue[str] = queue.Queue()
        self.debut = 0.0
        self.code: int | None = None
        self.titre = ""

    @property
    def en_cours(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def lancer(self, args: list[str], titre: str = "", env: dict | None = None) -> None:
        if self.en_cours:
            raise RuntimeError(f"{self.nom} déjà en cours")
        self.lignes, self.file, self.code, self.titre = [], queue.Queue(), None, titre
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        self.proc = subprocess.Popen([sys.executable, "-u", *args], cwd=RACINE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env={**os.environ, **(env or {}),
                                                                    "PYTHONIOENCODING": "utf-8"},
                                     **options)
        self.debut = time.time()
        threading.Thread(target=self._lire, daemon=True).start()

    def _lire(self) -> None:
        for brut in self.proc.stdout:
            ligne = brut.decode("utf-8", "replace").rstrip("\r\n")
            self.lignes.append(ligne)
            self.file.put(ligne)
        self.code = self.proc.wait()
        self.file.put(f"[fin, code {self.code}]")

    def arreter(self) -> None:
        if not self.en_cours:
            return
        if os.name == "nt":  # arrête aussi les processus de calcul lancés par la tâche
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            os.killpg(self.proc.pid, signal.SIGTERM)

    def nouvelles_lignes(self) -> list[str]:
        sortie = []
        while True:
            try:
                sortie.append(self.file.get_nowait())
            except queue.Empty:
                return sortie


TACHES = {"concepteur": Tache("IA Concepteur"), "optimisation": Tache("Optimisation"),
          "apprentissage": Tache("Apprentissage")}
