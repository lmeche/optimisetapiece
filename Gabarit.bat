@echo off
rem Lance Gabarit : l'application s'ouvre dans le navigateur (fermer cette fenetre arrete Gabarit).
cd /d "%~dp0"
rem Installe les bibliotheques ajoutees par une mise a jour (rien a faire si tout est deja la)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
".venv\Scripts\python.exe" -m gabarit.app %*
