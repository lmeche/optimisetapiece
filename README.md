# Gabarit

Conception assistée par IA d'ensembles mécano-soudés : une IA propose des concepts paramétriques
à partir d'un cahier des charges, un solveur éléments finis (Gmsh + CalculiX, en coques) les évalue
et les optimise (statique + flambement), puis une IA analyse les résultats et rédige le compte rendu.

Plan d'action : voir le document « Gabarit : plan d'action ».

## État

- [x] Phase 1 : chaîne Gmsh + CalculiX validée contre la RDM (flexion, contrainte, flambement d'Euler)
- [x] Phase 0 : cahier des charges structuré de la SAE 5.1 (`cahiers/sae51_equerre.json`, cotes relevées dans les STEP du socle et de l'adaptateur)
- [ ] Phase 2 et suivantes

## Installation (Windows)

1. Installer Python 3.11 ou 3.12 (python.org, cocher « Add to PATH »).
2. Installer FreeCAD (fournit `ccx.exe`, le solveur CalculiX) : `winget install FreeCAD.FreeCAD`.
3. Dans ce dossier : `powershell -ExecutionPolicy Bypass -File installer_windows.ps1`

Le script crée l'environnement `.venv`, repère `ccx.exe` et lance la validation.

## Validation du solveur

`python tests/valider_solveur.py` : tube carré 40 × 40 × 2 mm, 1000 mm, encastré.

| Grandeur | CalculiX | Théorie | Écart |
| --- | --- | --- | --- |
| Flèche en bout sous 1000 N | 21,756 mm | 21,777 mm | -0,10 % |
| Contrainte à mi-longueur | 137,3 MPa | 136,7 MPa | +0,46 % |
| Charge critique de flambement | 37 773 N | 37 910 N (Euler) | -0,36 % |
