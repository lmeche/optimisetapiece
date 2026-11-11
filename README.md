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
- [x] Version 2 : mesures fiables (garde-fou du solveur, extrémités de jonction), compte rendu chiffré, évidements, optimisation topologique, carnet d'apprentissage, IA Analyste et boucle d'apprentissage (`exemples/apprendre.py`)
- [x] Savoir physique : neuf principes de conception des tôles, outil de prédimensionnement de l'IA, prédictions physiques vérifiées par le calcul et calibrées dans le carnet (`gabarit/savoir/`, `exemples/predire.py`)
- [ ] Suite : introduction de l'effort par contact avec l'adaptateur, vérification non linéaire (plasticité, grands déplacements) des meilleurs concepts, installeur .exe, recoupement CATIA

## Phase 2 : ce que fait une évaluation (`gabarit/evaluation.py`)

1. Vérification des règles de fabrication (épaisseur atelier, encombrement, aile minimale de pliage, trous utilisables).
2. Maillage coque (Gmsh, quadrangles 8 noeuds) des surfaces moyennes.
3. Vis M10 : noeuds encastrés dans un rayon de 8 mm autour de chaque trou utilisé.
4. Contact du dos sur le socle : ensembles actifs (on bloque en X les noeuds qui pénètrent, on relâche ceux qui tirent).
5. Effort de 9 810 N réparti sur le disque Ø50 de l'adaptateur.
6. Statique : von Mises, flèche, traction par vis. Contrainte « utile » = hors singularités (anneau de 16 mm autour des vis, disque de 30 mm sous l'appui, boule de 10 mm autour de chaque extrémité de jonction : bout d'une soudure en T, d'un retour ou d'un doublage, où la contrainte d'un modèle coque croît sans fin quand on affine le maillage).
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

| Optimum (tôles 5 + 5 mm) | Maille 8 mm (exploration) | 6 mm (fiches du carnet) | 4 mm |
| --- | --- | --- | --- |
| Masse | 4,38 kg | 4,38 kg | 4,38 kg |
| Contrainte utile sous 9 810 N | 138 MPa | 146 MPa | 161 MPa |
| Effort tenu avant plastification | 16,7 kN | 15,8 kN | 14,3 kN |
| Facteur de flambement | 3,06 | 3,04 | 3,01 |
| Traction maxi par vis | 10,9 kN | 11,0 kN | 10,8 kN |
| Masse / effort tenu | 0,263 kg/kN | 0,277 kg/kN | 0,306 kg/kN |

Géométrie : largeur 148 mm, dessus à z = 192 mm, dos de z = 97 à 192 avec retours de 33 mm sur ses
bords, vis dans les colonnes y = ±60, V de 102 mm de large, pied à z = 59, diagonale soudée sous le
dessus en x = 313 (juste avant l'appui). La contrainte de flambement (>= 3) est celle qui limite :
il reste de la marge en plastification (x1,6 à 6 mm).

La contrainte utile augmente encore quand la maille s'affine, parce qu'elle est prise dans la zone
d'introduction de l'effort : au bout libre du dessus, à 5 mm du disque d'appui, et dans la diagonale juste
sous l'appui (6 mm), puis sur la jonction de la diagonale sous l'appui (4 mm). Au-delà de 50 mm de l'axe de la
charge, elle ne dépasse pas 111 MPa (96 MPa dans le dos). Le chiffre absolu dépend donc de la façon dont
l'effort entre dans la pièce et de sa modélisation (efforts égaux sur les noeuds du disque Ø50) : les concepts
se comparent à maille égale (6 mm pour les fiches du carnet), et le compte rendu de chaque concept dit si son
point critique est dans cette zone. Tableau refait avec la règle actuelle de la contrainte utile et le
garde-fou du solveur (voir la version 2) ; l'ancien donnait 154 et 183 MPa à 6 et 4 mm, lus au bout du retour
du dos, un point singulier.

Seconde passe sur tôles de 2 à 4 mm (`concepts/sae51_L_V_mince.json`, 128 essais) : aucun L + V
faisable, le meilleur tient 8,1 kN (4 + 4 mm, 1,6 kg, version étroite). Repris à la main en 5 + 5 mm,
ce L + V étroit pèse 2,0 kg et tient 12,4 kN, mais flambe à 1,1 x 1 t ; des retours de 28 mm sur la
diagonale ne l'aident pas (ils créent de nouvelles concentrations à leurs coins). Piste pour la suite :
un raidissement contre le flambement de la version étroite, qui gagnerait plus de 2 kg. (Calculs faits
avant l'exclusion des extrémités de jonction et le garde-fou du solveur : chiffres à refaire.)

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
cache du prompt. Une réponse coupée par la limite de longueur (64 000 jetons) n'est pas exécutée : le même
tour est relancé avec une limite doublée, jusqu'au maximum du modèle (128 000).

### Avec un modèle open source (`gabarit/ia/ouvert.py`)

Les outils de contrôle ne dépendent pas du fournisseur d'IA. Le moteur « ouvert » parle à toute API
compatible OpenAI, en local et gratuit (Ollama, LM Studio) ou chez un hébergeur de modèles ouverts
(OpenRouter, Mistral, Together) :

    python exemples/concevoir.py 2 --moteur ouvert --url http://localhost:11434/v1 --modele qwen3:8b

Ou dans l'écran Concepts du logiciel : bouton « Modèle ouvert », adresse, modèle et clé éventuelle
(variable `GABARIT_IA_CLE`). Si le modèle sait appeler des outils, il utilise `verifier_concept` et
`enregistrer_concept` comme Claude ; sinon (ou avec `--sans-outils`) il écrit le concept dans un bloc
```` ```json ````, Gabarit le contrôle, l'enregistre s'il est valide et lui renvoie le rapport.

| Serveur | Adresse |
| --- | --- |
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Mistral | `https://api.mistral.ai/v1` |

Le prompt fait environ 8 000 jetons : avec Ollama, agrandir la fenêtre de contexte
(`OLLAMA_CONTEXT_LENGTH=32768` avant de lancer Ollama). Un modèle de 7 à 8 milliards de paramètres
(ce que tient une carte graphique de 8 Go) produit des concepts plus simples et se trompe plus
souvent sur le format ; les contrôles l'empêchent d'enregistrer un concept invalide.

### Exploration longue (`gabarit/ia/exploration.py`)

Pour explorer beaucoup de formes sans payer de jetons : un petit modèle local tourne pendant des heures
(une nuit) et enchaîne des essais courts. Chaque essai repart d'une conversation neuve (le contexte d'un petit
modèle reste court), vise un seul concept sur une piste donnée (joues dégressives, âme centrale, caisson
fermé, cadre fermé, tirant suspendu... douze principes, puis des pistes inventées par le modèle ou des
reprises de pistes ratées), et reçoit la liste de ce qui a déjà été essayé pour ne pas le refaire.

    python exemples/explorer.py --heures 8 --url http://localhost:1234/v1 --modele qwen3-8b

Ou dans l'écran Concepts : « Modèle ouvert », case « Exploration longue » et durée. Les concepts valides
arrivent dans `concepts/exploration/` ; le catalogue de tous les essais (piste, résultat, masse, effort tenu
et kg/kN au calcul rapide, durée) dans `resultats/exploration/<date>/catalogue.csv`, aussi affiché sous
l'IA Concepteur. Le calcul rapide se fait au point central des variables : il sert à trier les pistes, les
meilleures passent ensuite par l'optimiseur. Régler la fenêtre de contexte du modèle à 16 000 jetons au moins
(LM Studio : au chargement du modèle ; Ollama : `OLLAMA_CONTEXT_LENGTH`).

Tests hors ligne (sans clé ni réseau, faux serveur local) : `python tests/tester_concepteur.py`,
`python tests/tester_concepteur_ouvert.py` et `python tests/tester_exploration.py`.

### Le concept « caisson galbé » (`concepts/sae51_caisson_galbe.json`)

Idée d'Iwan (une joue triangulaire sous un dessus) reprise en caisson fermé de deux tôles : tôle A pliée (dos vissé,
dessus en quatre facettes, face d'appui), tôle B pliée en U (deux joues, semelle droite qui remonte du pied du dos
jusque sous l'appui). 14 variables : épaisseurs, largeur, colonnes de vis, hauteurs du dos et de l'appui, pied de la
semelle, début de l'appui, bout de la semelle, trois hauteurs du profil (0,5 à 2 fois le profil droit), poches des
joues et du dos.

Optimisation (200 essais, 103 faisables, 1 h 45 sur 4 coeurs) : tôles 4 + 2 mm, largeur 68 mm (vis en y = ±20),
dessus plat à z = 200, semelle de z = 20 au dos jusqu'à x = 330, juste après l'axe de la charge ; profil droit,
sans poche. Masse 2,34 kg.

| Maille | 8 mm | 6 mm (fiche F007) | 4 mm |
| --- | --- | --- | --- |
| Caisson | 0,198 kg/kN | 0,235 kg/kN (10,0 kN tenus) | 0,204 kg/kN |
| L + V | 0,263 kg/kN | 0,277 kg/kN | 0,306 kg/kN |

Le caisson gagne à chaque maille (15 à 33 %). La semelle, tenue sur ses deux bords, ne flambe plus (facteur 9,9) ;
ce qui limite est la flexion du dos au coin haut. Des poches dans les joues ne paient pas (−1 % au mieux). Leçon L010
au carnet ; prédiction PR003 juste (`apprentissage/predictions/caisson_galbe.json`).

## Version 2 : apprendre et alléger

But : que l'IA comprenne pourquoi un concept tient plus de kN par kg qu'un autre, qu'elle s'en serve pour
concevoir les suivants, et que les tôles puissent être évidées. Ce que l'IA apprend repose sur des mesures de
Gabarit : rien de ce qu'elle écrit n'est pris pour une mesure.

### Des mesures fiables d'abord

- **Garde-fou du solveur** (`gabarit/fem/calculix.py`) : chaque calcul statique vérifie son bilan d'énergie
  (énergie des éléments = demi-travail des efforts, à 5 % près) et l'équilibre des réactions (à 0,1 % près ;
  une solution saine donne 1e-7). Une solution fausse est recalculée (3 essais), un mécanisme est refusé.
  Mesuré sur L + V : en 4 tâches, le solveur SPOOLES rend 3 solutions fausses sur 40 (équilibre faux de 4 à
  18 %, énergie jusqu'à +1 563 %), aucune en 1 tâche. Une tâche est donc le défaut (variable
  `GABARIT_THREADS`), pour un temps à peine plus long. Les vérifications à 6 et 4 mm faites avant ce
  garde-fou (en 4 tâches) peuvent être fausses : celles du tableau L + V ci-dessus ont été refaites.
- **Extrémités de jonction** (point 6 de la phase 2) : sur L + V, la contrainte au bout du retour du dos valait
  135, 154, 162 puis 183 MPa à 8, 6, 5 et 4 mm, sans converger. Avec l'exclusion, la contrainte utile vaut
  157, 138, 146, 149 et 161 MPa à 10, 8, 6, 5 et 4 mm. Ce qui dérive encore vient de la zone d'introduction
  de l'effort (voir le tableau L + V) : le compte rendu le signale pour chaque concept, et un écart de quelques %
  entre deux concepts reste du bruit, d'où le seuil de 3 % des expériences et l'estimation du bruit du maillage
  dans le carnet.
- **Affinage local** (`gabarit/optimisation.py`) : COBYLA demande au moins n + 2 évaluations ; il est maintenant
  arrêté au budget fixé, et son point de départ, déjà calculé, n'est pas refait (contrôle dans
  `tests/valider_optimiseur.py`).

### Compte rendu chiffré (`gabarit/analyse/compte_rendu.py`)

Identité exacte : **kg/kN = 1000 rho / Re x L_eq / eta**.
- L_eq (mm), longueur équivalente du chemin d'effort = somme sur la matière de V x contrainte / F : petite quand
  l'effort va droit aux fixations en traction ou compression, grande avec des détours ou de la flexion.
- eta, utilisation de la matière = contrainte moyenne / contrainte au point critique : 1 si toute la matière
  travaillait autant que le point le plus chargé, 0,5 au plus pour une tôle en flexion pure.

Le rapport des kg/kN de deux concepts est donc exactement le rapport des L_eq divisé par celui des eta. Le
compte rendu donne aussi, pour chaque face, sa part de masse, d'énergie et de L_eq, son eta, sa part de flexion,
la part de sa surface sous 20 % de la contrainte utile (matière peu utile) et une carte d'utilisation en texte ;
le point critique situé (face, bord, soudure, appui) avec sa distance à l'axe de la charge et la contrainte
maximale au-delà de 50 mm de cet axe (la zone d'appui fixe-t-elle l'effort tenu ?), la limite (plastification,
flambement, vis) et les marges, les faces du mode de flambement, des pistes automatiques. Validé sur des cas
analytiques (flexion, traction, Euler) et sur l'identité exacte : `python tests/valider_analyse.py` (19 contrôles).

### Évidements (`gabarit/concepts/evidements.py`)

Poches qui suivent le contour d'une face (« decalage », avec nervures possibles), trous ronds et découpes
polygonales, déclarés dans une tôle comme un retour. La variable « evidement » laisse l'optimiseur décider si la
poche vaut sa masse et régler la largeur de la bande laissée au bord. Zones protégées (plis, soudures, vis,
appui), bandes de matière d'au moins max(2 t, 10 mm), coins arrondis d'au moins max(t, 5 mm), maillage raffiné
au bord des trous. Validation (`python tests/valider_evidements.py`) : plaque trouée en traction, contrainte au
bord du trou à +0,06 % de la valeur de Peterson (Kt 2,349), masse à 0,003 % près.

### Optimisation topologique (`gabarit/analyse/topologie.py`)

`python exemples/alleger.py concepts/sae51_L_V.json --valeurs tests/donnees/sae51_L_V_optimum.json`

Méthode ESO (Xie et Steven, 1993), en « tout ou rien » sur le maillage coque : à chaque itération, calcul
statique puis retrait de 5 % de la matière évidable la moins sollicitée (contrainte de peau filtrée sur deux
mailles pour éviter les damiers), hors zones protégées et bandes de bord. L'ESO ne fait que classer la
matière : le bord en escalier fausse le kg/kN des itérations. Aux niveaux 15, 30, 45 et 60 %, la matière
retirée devient des poches fabricables (coins arrondis au rayon de fabrication, parties trop étroites
supprimées), et chaque candidat est recalculé par la chaîne complète (maillage raffiné au bord des trous,
flambement confirmé sur deux maillages) : c'est ce calcul qui décide. Une face qui participe au premier mode de
flambement n'est pas évidée quand la marge au flambement est faible.

Sur L + V à son optimum (maille 6 mm), aucune poche ne paie, et chaque candidat dit pourquoi
(kg/kN = masse x contrainte utile, et où est passé le point critique) :

| Candidat | Poche dans le dessus | Masse | Contrainte utile | kg/kN | Flambement |
| --- | --- | --- | --- | --- | --- |
| Sans poche | - | 4,38 kg | 146 MPa | 0,277 | 3,04 |
| ESO 15 % | 46 cm², au milieu | 4,20 kg (-4 %) | 157 MPa (+7 %) | 0,285 (+3 %) | 3,02 |
| ESO 30 % | 99 cm² | 3,99 kg (-9 %) | 247 MPa | 0,428, non tenu | 3,02 |
| ESO 45 % | 154 cm² | 3,77 kg (-14 %) | 254 MPa | 0,416, non tenu | 2,98 |

Le milieu du dessus, peu contraint, transmet pourtant la traction du dessus au dos : une poche la renvoie dans
les bandes latérales et vers la zone d'appui, qui fixe l'effort tenu. Le dos et la jambe, qui travaillent peu,
n'offrent rien à évider une fois les zones protégées retirées (vis, plis, soudures, bandes de bord). Le calcul
du niveau 62 % a échoué (CalculiX). La première ESO, faite avant l'exclusion des extrémités de jonction,
annonçait un kg/kN x1,36 pour une poche voisine : ce pic était celui du bout du retour du dos ; recalculée avec
la règle actuelle, cette poche (fiche F002 du carnet) donne -1 %, dans le bruit.

### Le carnet, l'IA Analyste et la boucle (`gabarit/apprentissage/`)

**Le carnet** (`apprentissage/carnet.json`, et `carnet.md` lisible) garde trois sortes d'entrées :
- des **fiches** : chaque concept mesuré par la chaîne complète à 6 mm avec flambement (kg/kN, L_eq, eta, limite,
  marges, point critique, bilan de chaque face, caractéristiques comme le nombre de tôles, de soudures ou de
  retours), avec le kg/kN de l'optimiseur à 8 mm comme mesure de convergence ;
- des **leçons** : un énoncé, son mécanisme, sa portée et ses preuves. Une IA désigne deux fiches, une grandeur et
  le sens attendu ; Gabarit lit lui-même les valeurs et tranche (seuil 3 %) ;
- des **expériences** : modifier UNE chose d'un concept, avec la prédiction (grandeur, sens) écrite et enregistrée
  avant le calcul. Une fois l'enfant mesuré, Gabarit rend le verdict : confirmée, contredite ou non concluante.
  Une pièce nettement non faisable compte comme moins bonne quel que soit son kg/kN. Un écart de kg/kN ne compte
  que s'il dépasse le seuil dans le même sens aux deux mailles (6 mm pour la fiche, 8 mm pour l'optimiseur) :
  sinon il dépend de la maille, et l'expérience est non concluante.

Statut d'une leçon : *hypothèse* (rien ne l'étaye), *observée* (des comparaisons vont dans son sens ; deux
concepts diffèrent toujours par plusieurs choses, une comparaison ne prouve pas une cause), *confirmée* ou
*contredite* (par au moins une expérience), *nuancée* (expériences dans les deux sens). Statistiques sur toutes
les fiches : part des écarts de kg/kN due au chemin d'effort (L_eq) et à l'utilisation de la matière (eta)
(décomposition exacte de la variance de ln kg/kN), liens entre les caractéristiques et le kg/kN (Spearman, valeur
p par permutation exacte jusqu'à 8 concepts, corrigée de Holm pour les tests multiples), bruit du maillage
(décalage du kg/kN entre 8 et 6 mm d'un concept à l'autre).

**L'IA Analyste** (`gabarit/apprentissage/analyste.py`, Claude ou modèle ouvert) reçoit les fiches nouvelles,
leurs références et les décompositions exactes calculées par Gabarit, et répond en JSON imposé : une explication
des écarts, des leçons (nouvelles, précisées ou appuyées, avec leurs comparaisons) et des expériences. Tout est
contrôlé avant d'entrer dans le carnet : fiches et grandeurs connues, nombres de l'IA ignorés, leçon nouvelle
refusée si toutes ses comparaisons vont contre les mesures.

**La boucle** (`gabarit/apprentissage/boucle.py`), un cycle à la fois :
1. choisir : tester la plus ancienne expérience prévue, sinon améliorer le meilleur concept (une modification,
   prédiction écrite dans le concept), et un cycle sur trois explorer un principe différent de tous ceux du
   carnet ;
2. l'IA Concepteur écrit le concept avec les leçons du carnet dans son prompt (une leçon que des constats
   contredisent y est signalée). Pendant une expérience, son outil de contrôle ne fait pas de calcul rapide : la
   prédiction se fait à l'aveugle. Hors de la boucle (`exemples/concevoir.py`, l'appli, l'exploration), l'IA
   Concepteur lit aussi la version courte du carnet ;
3. mini-optimisation : un enfant part des cotes optimisées de son parent (son premier essai est le parent
   modifié, 16 + 32 + 16 essais ensuite), un concept exploré part de rien avec un budget plus grand
   (32 + 64 + 24) ; budget et départ sont notés dans la fiche ;
4. fiche à 6 mm avec flambement, verdict de l'expérience, analyse, carnet enregistré après chaque étape.

```
python exemples/apprendre.py --cycles 3                       # Claude conçoit et analyse (ANTHROPIC_API_KEY)
python exemples/apprendre.py --cycles 20 --moteur ouvert --url http://localhost:1234/v1 --modele qwen3-8b
python exemples/apprendre.py --cycles 20 --moteur ouvert --analyste claude   # formes gratuites, analyse par Claude
python exemples/apprendre.py --afficher                       # le carnet, lisible
python exemples/apprendre.py --analyser F001 F002             # l'IA Analyste relit des fiches, sans calcul
```

Un cycle prend de 5 à 20 minutes ; le journal de chaque série est dans `resultats/apprentissage/<date>/journal.md`.
Avec un modèle ouvert comme concepteur, l'analyse par Claude ne coûte qu'une requête par cycle. Essai réel de la
reprise sur L + V (8 mm) : le premier essai redonne exactement le kg/kN du parent (0,2625), et l'affinage descend
ensuite à 0,2578 kg/kN.

**Le carnet de départ** (`python exemples/amorcer_carnet.py`, environ 8 minutes, refait tout depuis zéro) :
- quatre fiches recalculées par la chaîne actuelle : L + V à l'optimum de la V1 (F001, 0,277 kg/kN) et trois
  variantes à poche tirées de l'ESO (F002 à F004 : 0,274, 0,285 et 0,43 kg/kN, la dernière non tenue) ; leurs
  comptes rendus sont dans `apprentissage/comptes_rendus/` ;
- huit leçons avec leur provenance : constats des phases 2 à 4 notés comme tels (faits avant le carnet, ou avant
  l'exclusion des extrémités de jonction), constats lus dans les fiches, comparaisons dont Gabarit lit lui-même
  les valeurs. La plus utile : sur L + V, c'est la zone d'introduction de l'effort qui fixe l'effort tenu ;
- une expérience prévue, E001, que le premier cycle réalisera : prolonger le haut de la diagonale par une patte
  pliée à plat sous le dessus, sous l'axe de la charge, pour étaler l'effort de l'appui (prédiction écrite
  d'avance : kg/kN en baisse d'au moins 3 %).

Ses statistiques donnent déjà le bruit du maillage : d'un concept à l'autre, le décalage du kg/kN entre 8 et
6 mm varie de 4,4 % (écart-type sur les quatre fiches), si bien que deux concepts à moins d'environ 13 % l'un de
l'autre peuvent s'inverser d'une maille à l'autre. C'est la raison de la règle des deux mailles ci-dessus.

Tests hors ligne (sans CalculiX, sans clé ni réseau) : `python tests/tester_apprentissage.py` (statistiques
comparées à SciPy, verdicts, réponses d'IA piégées, faux serveurs Claude et modèle ouvert, boucle complète avec
des doublures des étapes coûteuses) et `python tests/tester_topologie.py`.

### Savoir physique (`gabarit/savoir/`)

Jusqu'ici l'IA Concepteur recevait le cahier des charges, le format des concepts, les règles de fabrication et
les leçons du carnet, mais aucune physique : elle dessinait presque au hasard et les contrôles triaient après
coup. Le savoir physique lui apprend comment une tôle porte un effort, sans toucher au cahier des charges (il dit
ce qu'on exige ; le savoir dit comment on conçoit, et il se vérifie).

- **Neuf principes** (`principes.py`), chacun avec sa règle, son mécanisme, sa formule, son domaine de validité
  et la grandeur que Gabarit mesure pour le vérifier : P1 un couple plutôt que de la flexion (tirant et bielle
  écartés de h), P2 travail dans le plan, P3 comprimé veut dire plié (flambement, voilement, b/t), P4 un appui
  sous la charge, P5 la compression passe par contact, P6 pas de matière oisive (eta), P7 l'épaisseur suit le
  mode de ruine, P8 des liaisons continues, P9 le matériau réel (écrouissage, réserve plastique). Le prompt les
  reçoit avec les chiffres du cas, calculés depuis le cahier des charges (moment 2,94 kN.m, h = 170 mm entre la
  semelle et les vis, borne du treillis 0,041 kg/kN, b/t maximal 42 et 14 en S235).
- **L'outil `predimensionner`** (`outil.py`, formules dans `calculs.py`) : une calculatrice sans éléments finis,
  à réponse immédiate. L'IA lui donne une hauteur de couple, des bielles et leur section (plaque ou U), des parois
  comprimées, des soudures ; il rend efforts, contraintes, facteurs de flambement et de voilement, masse, borne du
  kg/kN, écrouissage des plis, gorge de soudure, et ce qui coince avec le principe en jeu. Un petit modèle calcule
  mal de tête : l'outil fait l'arithmétique. Le modèle ouvert sans outils l'appelle par un bloc
  ```` ```json {"predimensionner": {...}} ````. Ses chiffres retrouvent ceux du L + V : avec K = 0,575 (barre
  soudée aux deux bouts), la diagonale plate a un facteur de flambement de 3,04, celui du calcul complet.
- **La prédiction physique** (`prediction.py`) : chaque concept porte un champ « prediction » écrit avant tout
  calcul (kg/kN attendu, mode de ruine, face du point critique en option, principes cités). Au calcul de la
  fiche, Gabarit la juge : juste si le kg/kN est à 15 % près et le mode exact, partielle si un seul l'est, fausse
  sinon ; le lieu est jugé à part. Le carnet garde les prédictions (PR001...) et, pour chaque principe, le compte
  de ses prédictions justes et le biais de l'écart (calibration), repris dans le prompt. Dans la boucle, la
  prédiction est enregistrée avant l'optimisation ; dans l'exploration de nuit, elle va au catalogue.

Pour tester une idée à la main, comme l'IA : écrire l'essai et sa prédiction dans un fichier, puis

```
python exemples/predire.py apprentissage/predictions/lv_diag_u2.json
```

enregistre la prédiction dans le carnet, calcule la fiche (6 mm avec flambement, contrôle à 8 mm) et rend le
verdict (environ 2 minutes).

**Premier essai, sur la diagonale du L + V** (fiches F005 et F006, prédictions PR001 et PR002) : plier la
diagonale en U de 2 ou 3 mm règle bien son flambement (facteur 3,43 mesuré pour 3,53 prévu en 2 mm), et la pièce
devient bien plus lourde par kN, comme prévu. Mais les deux prédictions étaient trop optimistes (écarts de -23 et
-37 % sur le kg/kN) et le mécanisme supposé était faux : le point critique n'est pas sous l'appui mais au pied de
la bielle, dans le pli vers la jambe où s'arrêtent les retours (344 et 274 MPa). D'où la leçon L009 et
l'expérience E004 (renforcer le pied d'une bielle en U de 3 mm), que la boucle testera. Les preuves que ces deux
essais apportaient à L007 ont été jugées non concluantes, puisque le mécanisme n'était pas le sien.

Le raisonnement complet, avec les chiffres, est dans le document « Gabarit : donner à l'IA le sens physique des
tôles ». Tests hors ligne : `python tests/tester_savoir.py` (formules recoupées avec ce document et la fiche F001,
outil, prédictions, calibration, prompt, mode texte).

## Le logiciel (`gabarit/app/`)

Application locale : elle tourne sur le PC (calculs compris) et s'affiche dans le navigateur.
Après `installer_windows.ps1`, double-cliquer sur le raccourci Gabarit du bureau (ou `Gabarit.bat`,
ou `python -m gabarit.app` ; `--fenetre` pour une fenêtre native si `pywebview` est installé). Le logiciel
n'est visible que depuis ce PC ; `--reseau` l'ouvre aux autres appareils du réseau local (sans mot de passe :
à éviter sur un réseau partagé, la clé API serait utilisable par les autres).

| Écran | Ce qu'on y fait |
| --- | --- |
| 1. Cahier des charges | Matériau, effort, critère de flambement, épaisseurs et vés de l'atelier, encombrement, grille de fixation, hypothèses ; enregistrement dans `cahiers/` |
| 2. Concepts | Liste des concepts (écrits à la main ou par l'IA), maquette 3D au point central ou sur un tirage aléatoire, variables, justifications, bouton « Vérifier » (géométrie + calcul rapide), lancement de l'IA Concepteur (Claude ou modèle ouvert) avec une consigne libre |
| 3. Optimisation | Choix du concept et du budget, suivi en direct (nuage des essais, meilleur résultat), arrêt ; une étude existante est reprise |
| 4. Résultats | Comparaison des concepts, optimum en 3D, vérification à 6 et 4 mm, valeurs des variables, Von Mises, bouton « Télécharger le STEP » |
| 5. Apprentissage | Lancer des cycles d'apprentissage (Claude ou modèle ouvert, IA Analyste au choix), suivre le carnet : fiches des concepts mesurés, leçons et leur statut, expériences et verdicts, statistiques |

La clé API (icône clé) reste sur le PC (`~/.gabarit/config.json` si on choisit de la mémoriser).
Les calculs longs tournent dans des processus séparés : Gmsh et CalculiX ne supportent pas les fils
d'exécution, et l'interface reste fluide.

Export STEP (`gabarit/export/step.py`) : chaque face devient une plaque de son épaisseur, plis vifs
sans rayon, une pièce par tôle ; les faces doublées sont décalées pour ne pas se chevaucher.

