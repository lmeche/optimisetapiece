@echo off
rem Lance Gabarit : l'application s'ouvre dans le navigateur (fermer cette fenetre arrete Gabarit).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m gabarit.app %*
