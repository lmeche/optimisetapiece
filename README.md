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
- [x] Phase 4 : format de concept déclaratif, concept « L + V » optimisé, IA Concepteur (`exemples/concevoir.py`)
- [x] Logiciel : prototype d'application locale (`Gabarit.bat` ou `python -m gabarit.app`)
- [ ] Phase 5 et suivantes : IA Analyste, compte rendu, installeur .exe, recoupement CATIA

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

## Phase 4 : concepts déclaratifs et IA Concepteur

### Le format (`gabarit/concepts/format.py`)

Un concept est un fichier JSON (exemple : `concepts/sae51_L_V.json`) : des variables classées
(épaisseur de l'atelier, choix discret, réel borné, fraction, retour optionnel), des dérivées
(expressions évaluées dans l'ordre), des tôles faites de faces planes (surfaces moyennes), des retours,
des soudures, la face de fixation, la face d'appui et les justifications de chaque choix.
Le même fichier sert à l'IA (qui l'écrit), à l'optimiseur (qui fait varier les variables) et au calcul.

- Deux faces d'une même tôle qui partagent une arête forment un pli : angle intérieur >= 30° et aile
  >= 0,7 x vé (vé standard le plus proche de 8 x t) vérifiés.
- Des tôles qui se touchent sont soudées (Gmsh les découpe ensemble, jonctions en T comprises) ; chaque
  soudure déclarée est contrôlée (arête commune d'au moins 10 mm ou doublage), l'assemblage doit être
  d'un seul tenant.
- Deux faces coplanaires superposées forment un doublage : les épaisseurs s'additionnent.
- Retour optionnel : 0 (pas de pli) ou une hauteur entre l'aile minimale et un maximum. L'optimiseur
  décide si le bord vaut sa masse (c'est l'idée du « pli à 0 mm partout »).

### Le concept « L + V »

Tôle en L en haut (dos vissé, dessus qui reçoit l'appui) et tôle en V dessous (jambe doublée contre
le dos, diagonale soudée sous le dessus) : dos, dessus et diagonale forment un triangle.
14 variables dont 4 retours optionnels (côtés et bout du dessus, bords de la diagonale, bords du dos).
Optimisation : `python exemples/optimiser_concept.py concepts/sae51_L_V.json 4`.

Résultat (168 essais, 46 faisables, 1 h sur 4 coeurs) : **le L + V tient 1 t**, l'équerre en U n'y arrivait pas.

| Optimum (tôles 5 + 5 mm) | Maille 8 mm (exploration) | 6 mm | 4 mm |
| --- | --- | --- | --- |
| Masse | 4,38 kg | 4,38 kg | 4,38 kg |
| Contrainte utile sous 9 810 N | 138 MPa | 154 MPa | 183 MPa |
| Effort tenu avant plastification | 16,7 kN | 15,0 kN | 12,6 kN |
| Facteur de flambement | 3,06 | 3,12 | 3,01 |
| Traction maxi par vis | 10,9 kN | 11,0 kN | 10,8 kN |
| Masse / effort tenu | 0,263 kg/kN | 0,292 kg/kN | 0,348 kg/kN |

Géométrie : largeur 148 mm, dessus à z = 192 mm, dos de z = 97 à 192 avec retours de 33 mm sur ses
bords, vis dans les colonnes y = ±60, V de 102 mm de large, pied à z = 59, diagonale soudée sous le
dessus en x = 313 (juste avant l'appui). La contrainte de flambement (>= 3) est celle qui limite :
il reste de la marge en plastification. La contrainte utile augmente quand la maille s'affine
(extrémités de la soudure diagonale / dessus) : on retient la valeur à 4 mm.

Seconde passe sur tôles de 2 à 4 mm (`concepts/sae51_L_V_mince.json`, 128 essais) : aucun L + V
faisable, le meilleur tient 8,1 kN (4 + 4 mm, 1,6 kg, version étroite). Repris à la main en 5 + 5 mm,
ce L + V étroit pèse 2,0 kg et tient 12,4 kN, mais flambe à 1,1 x 1 t ; des retours de 28 mm sur la
diagonale ne l'aident pas (ils créent de nouvelles concentrations à leurs coins). Piste pour la suite :
un raidissement contre le flambement de la version étroite, qui gagnerait plus de 2 kg.

### L'IA Concepteur (`gabarit/ia/`)

`python exemples/concevoir.py 3` (clé dans `ANTHROPIC_API_KEY`) : Claude reçoit le cahier des
charges, le format, les règles de fabrication, ce qu'on a appris des concepts précédents et le L + V en
exemple, puis propose des formes générales. Il contrôle lui-même chaque concept avec l'outil
`verifier_concept` (géométrie au point central et sur 16 tirages aléatoires, calcul rapide au
point central avec la zone la plus chargée), corrige, et n'enregistre (`enregistrer_concept`) qu'un
concept qui passe les contrôles. Les concepts arrivent dans `concepts/ia/`, le journal (raisonnement
résumé, appels, bilan) dans `resultats/concepteur/<date>/journal.md`.

Réglages : modèle `claude-opus-5-5`, réflexion adaptative, effort `high` (`--effort` pour changer),
relais automatique vers un modèle de secours si une requête est déclinée (`fallbacks: "default"`),
cache du prompt. Les outils de contrôle (`gabarit/ia/outils.py`) ne dépendent pas du fournisseur
d'IA : un modèle open source pourra s'y brancher.

Test hors ligne (sans clé ni réseau, faux serveur local) : `python tests/tester_concepteur.py`.

## Le logiciel (`gabarit/app/`)

Application locale : elle tourne sur le PC (calculs compris) et s'affiche dans le navigateur.
Après `installer_windows.ps1`, double-cliquer sur le raccourci Gabarit du bureau (ou `Gabarit.bat`,
ou `python -m gabarit.app` ; `--fenetre` pour une fenêtre native si `pywebview` est installé).

| Écran | Ce qu'on y fait |
| --- | --- |
| 1. Cahier des charges | Matériau, effort, critère de flambement, épaisseurs et vés de l'atelier, encombrement, grille de fixation, hypothèses ; enregistrement dans `cahiers/` |
| 2. Concepts | Liste des concepts (écrits à la main ou par l'IA), maquette 3D au point central ou sur un tirage aléatoire, variables, justifications, bouton « Vérifier » (géométrie + calcul rapide), lancement de l'IA Concepteur avec une consigne libre |
| 3. Optimisation | Choix du concept et du budget, suivi en direct (nuage des essais, meilleur résultat), arrêt ; une étude existante est reprise |
| 4. Résultats | Comparaison des concepts, optimum en 3D, vérification à 6 et 4 mm, valeurs des variables, Von Mises, bouton « Télécharger le STEP » |

La clé API (icône clé) reste sur le PC (`~/.gabarit/config.json` si on choisit de la mémoriser).
Les calculs longs tournent dans des processus séparés : Gmsh et CalculiX ne supportent pas les fils
d'exécution, et l'interface reste fluide.

Export STEP (`gabarit/export/step.py`) : chaque face devient une plaque de son épaisseur, plis vifs
sans rayon, une pièce par tôle ; les faces doublées sont décalées pour ne pas se chevaucher.

