# Gabarit

Conception assistée par IA d'ensembles mécano-soudés : une IA propose des concepts paramétriques
à partir d'un cahier des charges, un solveur éléments finis (Gmsh + CalculiX, en coques) les évalue
et les optimise (statique + flambement), puis une IA analyse les résultats et rédige le compte rendu.

Plan d'action : voir le document « Gabarit : plan d'action ».

## État

- [x] Phase 1 : chaîne Gmsh + CalculiX validée contre la RDM (flexion, contrainte, flambement d'Euler)
- [x] Phase 0 : cahier des charges structuré de la SAE 5.1 (`cahiers/sae51_equerre.json`, cotes relevées dans les STEP du socle et de l'adaptateur)
- [x] Phase 2 : équerre en U de référence calculée sur la vraie grille (`exemples/phase2_equerre_u.py`)
- [x] Phase 3 : moteur d'optimisation validé (`tests/valider_optimiseur.py`) et appliqué à l'équerre en U (`exemples/phase3_optimisation.py`)
- [ ] Phase 4 et suivantes

## Phase 2 : ce que fait une évaluation (`gabarit/evaluation.py`)

1. Vérification des règles de fabrication (épaisseur atelier, encombrement, aile minimale de pliage, trous utilisables).
2. Maillage coque (Gmsh, quadrangles 8 noeuds) des surfaces moyennes.
3. Vis M10 : noeuds encastrés dans un rayon de 8 mm autour de chaque trou utilisé.
4. Contact du dos sur le socle : ensembles actifs (on bloque en X les noeuds qui pénètrent, on relâche ceux qui tirent).
5. Effort de 9 810 N réparti sur le disque Ø50 de l'adaptateur.
6. Statique : von Mises, flèche, traction par vis. Contrainte « utile » = hors singularités (anneau de 16 mm autour des vis, disque de 30 mm sous l'appui).
7. Flambement linéaire : seuls les facteurs retrouvés à 3 % près sur deux maillages sont retenus (des modes parasites isolés apparaissent sinon).
8. Effort tenu = 9 810 x Re / contrainte utile ; objectif masse / effort tenu.

Convergence (équerre de référence) :

| Taille de maille | Contrainte utile | Flèche | Facteur de flambement | Temps |
| --- | --- | --- | --- | --- |
| 8 mm | 653 MPa | 2,10 mm | 4,98 | 23 s |
| 6 mm | 633 MPa | 1,98 mm | 5,03 | 37 s |
| 4 mm | 612 MPa | 1,97 mm | 5,02 | 90 s |


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

## Phase 3 : optimisation (`gabarit/optimisation.py`)

1. Plan d'expériences quasi aléatoire (Sobol) pour couvrir l'espace des variables.
2. Optimisation bayésienne sous contraintes (Optuna, TPE).
3. Affinage local COBYLA sur les variables continues depuis le meilleur point faisable.

Évaluations en parallèle (processus séparés), toutes enregistrées dans une base SQLite.

Validation (`python tests/valider_optimiseur.py`) : poutre rectangulaire de section minimale sous
contrainte de flexion, optimum analytique retrouvé à 0,01 % (graines 0 à 3).

Étude SAE (`gabarit/etudes/sae51_equerre_u.py`) : 7 variables (épaisseur, écart des vis, largeur,
hauteur du dessus, bas du dos, hauteurs des ailes), objectif masse / effort tenu, contraintes : pas de
plastification à 1 t, flambement >= 3, traction des vis <= 33,4 kN (M10 8.8).

Résultat : sur 2 x 96 essais, **aucune équerre en U ne tient 1 t**. L'effort tenu plafonne vers
4 à 5 kN à 5 mm d'épaisseur, quelle que soit la masse. La zone critique est le dessus, qui fléchit
localement sous l'appui Ø50 (environ 600 MPa sous 9 810 N, valeur stable de 8 à 3 mm de maille).
Ce concept d'une seule tôle a besoin d'un renfort local (deuxième pièce soudée), ce qui relève de la phase 4.
