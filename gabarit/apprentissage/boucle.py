"""Boucle d'apprentissage : concevoir, optimiser, mesurer, comprendre, recommencer, en tenant un carnet.

Un cycle :
1. choisit une opération :
   - tester : faire une expérience prévue par l'IA Analyste (modifier UNE chose d'un concept mesuré ; la
     prédiction est déjà enregistrée) ;
   - ameliorer : l'IA Concepteur modifie UNE chose du meilleur concept et écrit sa prédiction dans le concept
     (champ « experience »), enregistrée dans le carnet avant tout calcul ;
   - explorer : l'IA Concepteur propose un principe différent de tous ceux du carnet (un cycle sur
     `explorer_tous`, et tant que le carnet a moins de deux concepts faisables).
   Pendant une expérience (tester, ameliorer), l'outil de contrôle de l'IA Concepteur ne fait pas de calcul
   rapide : la prédiction est faite à l'aveugle, sans aucun résultat de calcul sur le concept modifié ;
2. l'IA Concepteur (Claude ou modèle ouvert) écrit le concept, avec les leçons du carnet dans son prompt ;
3. mini-optimisation à budget fixé. Un enfant part des cotes optimisées de son parent (mêmes noms de variables) :
   son premier essai est « le parent avec la modification », l'optimiseur ne peut que faire mieux ensuite. Un
   concept exploré n'a pas ce point de départ et reçoit un budget plus grand. Le budget et le départ sont
   notés dans la fiche ;
4. fiche : compte rendu complet à la maille des fiches (6 mm, avec flambement) ; le kg/kN de l'optimiseur
   (8 mm) est gardé comme mesure de convergence ;
5. verdicts : Gabarit compare les mesures du parent et de l'enfant (expérience, seuil 3 %), et la fiche à la
   prédiction physique du concept (champ « prediction », enregistrée avant tout calcul : gabarit/savoir) ;
6. l'IA Analyste explique les écarts, range des leçons (vérifiées sur les mesures) et propose des expériences ;
7. le carnet est enregistré (apprentissage/carnet.json, et carnet.md lisible) après chaque étape qui le
   change, et le cycle est raconté dans resultats/apprentissage/<date>/journal.md.

Les étapes coûteuses (IA, optimisation, mesure, analyse) sont des fonctions passées en paramètre : les tests
les remplacent par des doublures.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gabarit.apprentissage.analyste import (
    EXPERIENCES_MAX,
    BilanAnalyse,
    analyser_claude,
    analyser_ouvert,
    texte_fiche,
)
from gabarit.apprentissage.carnet import (
    RACINE,
    Carnet,
    Experience,
    Fiche,
    Prediction,
    grandeur_valide,
    valeur,
)
from gabarit.savoir import prediction as prediction_physique

CDC = RACINE / "cahiers" / "sae51_equerre.json"
MODELE_CLAUDE = "claude-opus-5-5"
MODELE_OUVERT = "qwen3:8b"
URL_OLLAMA = "http://localhost:11434/v1"
BUDGET_ENFANT = (16, 32, 16)  # essais initiaux, bayésiens, d'affinage ; le premier essai est le parent modifié
BUDGET_EXPLORATION = (32, 64, 24)  # un concept nouveau part de rien
TAILLE_FICHE = 6.0
ATTENTE_MAX = 4  # expériences en attente au-delà desquelles l'IA Analyste n'en propose plus
ECHECS_MAX = 3  # cycles de suite interrompus par l'IA (serveur arrêté, clé refusée...) avant d'abandonner


@dataclass
class Reglages:
    moteur: str = "claude"  # IA Concepteur : claude ou ouvert
    modele: str = ""  # vide : claude-opus-5-5 ou qwen3:8b selon le moteur
    analyste: str = ""  # IA Analyste : claude, ouvert ou aucun ; vide : le moteur du concepteur
    modele_analyste: str = ""  # vide : le modèle du concepteur si même moteur, sinon le modèle par défaut
    url: str = URL_OLLAMA  # modèle ouvert : adresse de l'API compatible OpenAI
    cle: str = ""  # modèle ouvert hébergé : clé du service
    effort: str = "high"  # Claude
    chemin_cdc: Path = CDC
    budget: tuple[int, int, int] = BUDGET_ENFANT
    budget_exploration: tuple[int, int, int] = BUDGET_EXPLORATION
    n_paralleles: int = 4
    taille_fiche: float = TAILLE_FICHE
    explorer_tous: int = 3
    sortie: Path = RACINE / "concepts" / "apprentissage"
    outils: bool = True  # modèle ouvert : appels d'outils (sinon concept rendu dans le texte)
    delai: float = 1800.0  # modèle ouvert : délai d'une réponse (s)
    tours: int = 12  # modèle ouvert : tours de correction d'une session
    graine: int = 0  # même graine pour tous les concepts : même traitement

    def moteur_analyste(self) -> str:
        return self.analyste or self.moteur

    def nom_modele(self) -> str:
        return self.modele or (MODELE_CLAUDE if self.moteur == "claude" else MODELE_OUVERT)

    def nom_modele_analyste(self) -> str:
        if self.modele_analyste:
            return self.modele_analyste
        if self.moteur_analyste() == self.moteur:
            return self.nom_modele()
        return MODELE_CLAUDE if self.moteur_analyste() == "claude" else MODELE_OUVERT


# ---------- choix de l'opération et consignes ----------
def choisir_operation(carnet: Carnet, numero: int, explorer_tous: int = 3) -> tuple[str, Experience | Fiche | None]:
    """Explorer un cycle sur `explorer_tous` et tant que moins de deux concepts sont faisables ; sinon la plus
    ancienne expérience en attente ; sinon améliorer le meilleur concept."""
    faisables = carnet.meilleures(2)
    if len(faisables) < 2 or (explorer_tous and numero % explorer_tous == 0):
        return "explorer", None
    a_faire = [e for e in carnet.experiences_a_faire() if carnet.a_fiche(e.parent)]
    if a_faire:
        return "tester", a_faire[0]
    return "ameliorer", faisables[0]


def chemin_fiche(f: Fiche) -> Path:
    p = Path(f.fichier)
    return p if p.is_absolute() else RACINE / p


def _concept_parent(f: Fiche) -> dict:
    """Le concept d'une fiche, sans sa prédiction éventuelle (une nouvelle sera écrite pour l'enfant)."""
    d = json.loads(chemin_fiche(f).read_text(encoding="utf-8"))
    d.pop("experience", None)
    return d


def _bloc_parent(carnet: Carnet, f: Fiche, definition: dict) -> str:
    valeurs = json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.valeurs.items()}, ensure_ascii=False)
    return (f"### Mesures du concept de départ (fiche {f.id}, calculées par Gabarit)\n{texte_fiche(f)}\n\n"
            f"### Valeurs optimisées du concept de départ\n{valeurs}\n\n"
            f"### Concept de départ ({f.fichier})\n```json\n{json.dumps(definition, indent=1, ensure_ascii=False)}\n```")


REGLES_ENFANT = (
    "- Pars du concept de départ ci-dessous et ne change que ce qu'exige la modification : une seule chose change, "
    "sinon on ne saura pas ce qui a joué.\n"
    "- Garde les noms des variables qui ne changent pas : l'optimiseur repartira des valeurs optimisées du parent "
    "(son premier essai sera le parent avec ta modification), puis réajustera toutes les cotes.\n"
    "- Donne au concept un nom qui dit la modification.\n"
    "- Pendant une expérience, verifier_concept ne contrôle que la géométrie (pas de calcul rapide) : la prédiction "
    "se fait avant tout calcul.")


def consigne_tester(carnet: Carnet, e: Experience, definition: dict) -> str:
    f = carnet.fiche(e.parent)
    lecon = ""
    if e.lecon and any(x.id == e.lecon for x in carnet.lecons):
        x = carnet.lecon(e.lecon)
        lecon = f" Elle teste la leçon {x.id} ({x.statut}) : {x.enonce}"
    return (f"Expérience {e.id} à réaliser.{lecon}\n"
            f"Modification à faire sur le concept de départ (fiche {f.id}, « {f.nom} »), et seulement celle-là : "
            f"{e.modification}\n"
            f"Prédiction déjà enregistrée : {e.grandeur} en {e.sens} d'au moins {e.seuil_pct:g} %. Ne remplis pas le "
            "champ « experience ».\n" + REGLES_ENFANT + "\n\n" + _bloc_parent(carnet, f, definition))


def consigne_ameliorer(carnet: Carnet, f: Fiche, definition: dict) -> str:
    return (f"Améliore le meilleur concept mesuré (fiche {f.id}, « {f.nom} ») par UNE seule modification : celle qui "
            "devrait le plus baisser son kg/kN d'après ses mesures (point critique, limite, matière peu utile, L_eq et "
            "eta de chaque face) et d'après les leçons du carnet. Ne refais pas une modification déjà essayée.\n"
            "Écris ta prédiction dans le champ « experience » du concept, avant tout calcul : "
            f'{{"parent": "{f.id}", "modification": "ce qui change, en une phrase", "grandeur": "kg_par_kN", '
            '"sens": "baisse", "lecon": "id de la leçon que tu appliques, ou vide"}. Tu peux prédire une autre '
            "grandeur (sigma_utile, flambement, l_eq_mm, eta...) si c'est elle que ta modification vise ; Gabarit "
            "tranchera sur les mesures, seuil 3 %.\n" + REGLES_ENFANT + "\n\n" + _bloc_parent(carnet, f, definition))


def consigne_explorer(carnet: Carnet, n_max: int = 15) -> str:
    texte = ("Propose un concept dont le principe de structure (chemin d'effort, nombre de tôles, type de "
             "raidissement) diffère de tous ceux déjà mesurés. Appuie-toi sur les leçons confirmées et évite ce que "
             "les expériences ont contredit.")
    if carnet.fiches:
        deja = [f"- {f.id} « {f.nom} » : {f.principe[:200] or 'principe non décrit'} ("
                + (f"{valeur(f, 'kg_par_kN'):.3f} kg/kN" if math.isfinite(valeur(f, "kg_par_kN")) else "non calculé")
                + ("" if f.faisable else ", non faisable") + ")"
                for f in carnet.fiches if not f.parent][-n_max:]
        texte += "\n\nPrincipes déjà mesurés (ne refais pas le même) :\n" + "\n".join(deja)
    return texte


def enregistrer_prediction(carnet: Carnet, definition: dict, parent: str, source: str) -> Experience:
    """Expérience tirée du champ « experience » d'un concept, avant tout calcul. Le parent est imposé (celui
    qu'on a donné à l'IA) ; une grandeur ou un sens illisible est remplacé par la prédiction par défaut d'une
    amélioration (kg/kN en baisse), et la source le dit."""
    x = definition.get("experience") if isinstance(definition.get("experience"), dict) else {}
    grandeur, sens = str(x.get("grandeur") or "").strip(), str(x.get("sens") or "").strip()
    defaut = []
    if not grandeur_valide(grandeur):
        defaut.append("grandeur")
        grandeur = "kg_par_kN"
    if sens not in ("baisse", "hausse"):
        defaut.append("sens")
        sens = "baisse"
    modification = str(x.get("modification") or "").strip() or str(definition.get("description", "")).strip()[:300]
    lecon = str(x.get("lecon") or "").strip() or None
    note = f" ({', '.join(defaut)} par défaut)" if defaut else ""
    return carnet.ajouter_experience(parent, modification or "modification non décrite", grandeur, sens, lecon,
                                     source=source + note)


def enregistrer_prediction_physique(carnet: Carnet, definition: dict, chemin: Path, source: str,
                                    experience: str | None = None) -> Prediction | None:
    """Prédiction physique du concept (champ « prediction »), enregistrée avant tout calcul ; None si le concept
    n'en a pas ou si elle n'a ni kg/kN ni mode lisibles. Ce qui était illisible est gardé dans la prédiction."""
    p, erreurs = prediction_physique.lire(definition)
    if p is None or "kg_par_kN" not in p or "mode_limite" not in p:
        return None
    return carnet.ajouter_prediction(p, chemin, str(definition.get("nom", "")), source, experience, erreurs)


# ---------- étapes coûteuses ----------
def optimiser_concept(chemin: str | Path, chemin_cdc: str | Path = CDC, budget: tuple[int, int, int] = BUDGET_ENFANT,
                      n_paralleles: int = 4, dossier: Path | None = None, depart: list[dict] | None = None,
                      graine: int = 0, afficher=print) -> dict:
    """Mini-optimisation d'un concept (plan initial, bayésien, affinage) à la maille d'exploration.
    depart : valeurs de concepts connus (le parent) imposées aux premiers essais, traduites dans les variables de
    ce concept (`parametres_depuis_valeurs`). Renvoie les valeurs du meilleur essai faisable, ou à défaut de
    l'essai le plus proche de la faisabilité (somme des dépassements des contraintes la plus petite)."""
    import optuna

    from gabarit.etudes.concept_json import (
        TAILLE_EXPLORATION,
        EtudeConcept,
        parametres_depuis_valeurs,
    )
    from gabarit.optimisation import faisable, meilleur_faisable, optimiser

    etude = EtudeConcept(chemin, chemin_cdc)
    dossier = Path(dossier or tempfile.mkdtemp(prefix="gabarit_opt_"))
    dossier.mkdir(parents=True, exist_ok=True)
    params = [parametres_depuis_valeurs(etude.definition, etude.cdc, v) for v in (depart or [])]
    t0 = time.time()
    meilleur = [math.inf]

    def journal(tr):
        if faisable(tr):
            meilleur[0] = min(meilleur[0], tr.value)
        afficher(f"[{time.time() - t0:5.0f} s] essai {tr.number:3d} {'faisable' if faisable(tr) else '-       '} "
                 f"{tr.value:8.4f}  meilleur {meilleur[0]:.4f}", flush=True)

    study = optimiser(etude.proposer, etude.evaluer, n_initial=budget[0], n_bayes=budget[1],
                      affinage=etude.affinage(), n_affinage=budget[2], n_paralleles=n_paralleles,
                      stockage=f"sqlite:///{dossier / 'etude.db'}", nom=Path(chemin).stem, graine=graine,
                      journal=journal, depart=params)
    sortie = {"essais": len(study.trials), "faisables": sum(faisable(t) for t in study.trials), "budget": list(budget),
              "taille": TAILLE_EXPLORATION, "graine": graine, "duree_s": round(time.time() - t0)}
    if params:
        variables = etude.definition["variables"]
        premier = study.trials[0]
        sortie["reprise"] = {"variables_reprises": [n for n in variables if n in params[0] or f"{n}_actif" in params[0]],
                             "variables": len(variables), "faisable": faisable(premier),
                             "kg_par_kN": premier.value if faisable(premier) else None}
    best = meilleur_faisable(study)
    if best is None:
        calcules = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value < 1e3
                    and "params_concept" in t.user_attrs]
        best = min(calcules, key=lambda t: (sum(max(0.0, c) for c in t.user_attrs.get("contraintes", [1e9])), t.value),
                   default=None)
    if best is None:
        sortie.update(valeurs=None, kg_par_kN=None, faisable=False)
    else:
        sortie.update(valeurs=best.user_attrs["params_concept"], kg_par_kN=float(best.value), faisable=faisable(best),
                      essai=best.number)
    (dossier / "optimisation.json").write_text(json.dumps(sortie, indent=1, ensure_ascii=False, default=float) + "\n",
                                               encoding="utf-8")
    return sortie


def mesurer(carnet: Carnet, chemin: str | Path, valeurs: dict, chemin_cdc: str | Path = CDC,
            taille: float = TAILLE_FICHE, dossier: Path | None = None, **fiche) -> Fiche:
    """Fiche d'un concept à des valeurs données : compte rendu complet (statique avec contact, flambement) à la
    maille des fiches. Un concept qui ne se construit pas ou ne se calcule pas a quand même sa fiche (erreurs)."""
    from gabarit.analyse.compte_rendu import CompteRendu, analyser
    from gabarit.cahier_des_charges import CahierDesCharges
    from gabarit.concepts.format import ConceptJSON

    cdc = CahierDesCharges.charger(chemin_cdc)
    nom = Path(chemin).stem
    concept = None
    try:
        concept = ConceptJSON.charger(chemin, valeurs, cdc)
        nom = concept.definition.get("nom", nom)
        erreurs = concept.verifier(cdc)
        cr = (CompteRendu(nom, dict(valeurs), taille, False, erreurs) if erreurs
              else analyser(concept, cdc, taille=taille, flambement=True, dossier=dossier))
    except (RuntimeError, ValueError, KeyError) as ex:  # maillage impossible, solution fausse, variable absente...
        cr = CompteRendu(nom, dict(valeurs), taille, False, [str(ex)[:300]])
    if dossier is not None:
        Path(dossier).mkdir(parents=True, exist_ok=True)
        if cr.valide:
            (Path(dossier) / "compte_rendu.md").write_text(cr.markdown() + "\n", encoding="utf-8")
    return carnet.ajouter_fiche(cr, concept, fichier=chemin, **fiche)


def amorcer(carnet: Carnet, chemin: str | Path, valeurs: dict, chemin_cdc: str | Path = CDC,
            taille: float = TAILLE_FICHE, convergence_taille: float | None = 8.0, dossier: Path | None = None,
            **fiche) -> Fiche:
    """Fiche d'un concept déjà optimisé hors de la boucle (par exemple avec exemples/optimiser_concept.py) :
    compte rendu à la maille des fiches et kg/kN à la maille de l'optimiseur (convergence)."""
    from gabarit.cahier_des_charges import CahierDesCharges
    from gabarit.concepts.format import ConceptJSON
    from gabarit.evaluation import evaluer

    convergence = dict(fiche.pop("convergence", {}) or {})
    if convergence_taille and not convergence:
        cdc = CahierDesCharges.charger(chemin_cdc)
        concept = ConceptJSON.charger(chemin, valeurs, cdc)
        if not concept.verifier(cdc):
            r = evaluer(concept, cdc, taille=convergence_taille, flambement=False)
            if r.valide:
                convergence = {"taille": convergence_taille, "kg_par_kN": float(r.masse_par_kN)}
    return mesurer(carnet, chemin, valeurs, chemin_cdc, taille, dossier, convergence=convergence, **fiche)


def concevoir_ia(reglages: Reglages, consigne: str, lecons: str, exemples: list[Path] | None, calcul: bool,
                 dossier: Path, afficher=print):
    """Une session d'IA Concepteur pour un concept (Claude ou modèle ouvert)."""
    if reglages.moteur == "claude":
        from gabarit.ia.concepteur import concevoir

        return concevoir(1, chemin_cdc=reglages.chemin_cdc, exemples=exemples, sortie=reglages.sortie,
                         dossier_journal=dossier, modele=reglages.nom_modele(), effort=reglages.effort,
                         consigne=consigne, calcul=calcul, lecons=lecons, afficher=afficher)
    from gabarit.ia.ouvert import concevoir_ouvert

    return concevoir_ouvert(1, url=reglages.url, modele=reglages.nom_modele(), cle=reglages.cle,
                            chemin_cdc=reglages.chemin_cdc, exemples=exemples, sortie=reglages.sortie,
                            dossier_journal=dossier, consigne=consigne, calcul=calcul, outils=reglages.outils,
                            delai=reglages.delai, tours_max=reglages.tours, lecons=lecons, afficher=afficher)


def analyser_ia(reglages: Reglages, carnet: Carnet, nouvelles: list[str], experiences_max: int = EXPERIENCES_MAX,
                afficher=print) -> BilanAnalyse | None:
    """L'IA Analyste relit les fiches `nouvelles` (None si l'analyse est désactivée)."""
    moteur = reglages.moteur_analyste()
    if moteur == "aucun" or not nouvelles:
        return None
    if moteur == "claude":
        return analyser_claude(carnet, nouvelles, modele=reglages.nom_modele_analyste(), effort=reglages.effort,
                               experiences_max=experiences_max, afficher=afficher)
    return analyser_ouvert(carnet, nouvelles, url=reglages.url, modele=reglages.nom_modele_analyste(), cle=reglages.cle,
                           delai=reglages.delai, compact=True, experiences_max=experiences_max, afficher=afficher)


def _optimiser(chemin, reglages: Reglages, depart, dossier, afficher):
    budget = reglages.budget if depart else reglages.budget_exploration
    return optimiser_concept(chemin, reglages.chemin_cdc, budget, reglages.n_paralleles, dossier, depart,
                             reglages.graine, afficher)


def _mesurer(carnet, chemin, valeurs, reglages: Reglages, dossier, **fiche):
    return mesurer(carnet, chemin, valeurs, reglages.chemin_cdc, reglages.taille_fiche, dossier, **fiche)


# ---------- la boucle ----------
def apprendre(cycles: int = 1, reglages: Reglages | None = None, carnet: Carnet | None = None,
              dossier: Path | None = None, afficher=print, concepteur: Callable = concevoir_ia,
              optimiseur: Callable = _optimiser, mesureur: Callable = _mesurer,
              analyste: Callable = analyser_ia) -> Carnet:
    """Enchaîne `cycles` cycles d'apprentissage et renvoie le carnet (enregistré après chaque étape).
    Un cycle interrompu par l'IA (serveur arrêté, réseau, quota) est perdu mais pas le carnet ; après
    ECHECS_MAX interruptions de suite, ou une clé refusée, la boucle s'arrête."""
    reglages = reglages or Reglages()
    carnet = carnet if carnet is not None else Carnet.charger()
    dossier = Path(dossier or RACINE / "resultats" / "apprentissage" / time.strftime("%Y%m%d_%H%M%S"))
    dossier.mkdir(parents=True, exist_ok=True)
    echecs = 0
    for _ in range(cycles):
        try:
            un_cycle(carnet, reglages, dossier, afficher, concepteur, optimiseur, mesureur, analyste)
            echecs = 0
        except Exception as ex:
            if not _erreur_ia(ex):
                raise
            echecs += 1
            carnet.noter(f"cycle {carnet.cycles + 1} interrompu par l'IA : {str(ex)[:300]}")
            carnet.enregistrer()
            _journal(dossier, f"**Cycle interrompu par l'IA** : {ex}")
            afficher(f"\n[IA injoignable] {ex}\n")
            if echecs >= ECHECS_MAX or _cle_refusee(ex):
                afficher("Boucle arrêtée : l'IA ne répond pas (ou la clé est refusée).")
                break
    afficher(f"\nCarnet : {carnet.chemin} ({len(carnet.fiches)} fiches, {len(carnet.lecons)} leçons, "
             f"{len(carnet.experiences)} expériences). Journal : {dossier / 'journal.md'}")
    return carnet


def un_cycle(carnet: Carnet, reglages: Reglages, dossier: Path, afficher=print, concepteur: Callable = concevoir_ia,
             optimiseur: Callable = _optimiser, mesureur: Callable = _mesurer, analyste: Callable = analyser_ia) -> dict:
    numero = carnet.cycles + 1
    operation, cible = choisir_operation(carnet, numero, reglages.explorer_tous)
    e = cible if operation == "tester" else None
    parent = carnet.fiche(e.parent) if e else (cible if operation == "ameliorer" else None)
    d = dossier / f"cycle_{numero:03d}_{operation}"
    trace = {"cycle": numero, "operation": operation, "parent": parent.id if parent else None,
             "experience": e.id if e else None}
    titre = {"tester": f"tester l'expérience {e.id if e else ''}", "ameliorer": f"améliorer {parent.id if parent else ''}",
             "explorer": "explorer un principe nouveau"}[operation]
    afficher(f"\n=== Cycle {numero} : {titre} ===\n")
    _journal(dossier, f"## Cycle {numero} : {titre}" + (f"\n\n{e.texte()}" if e else ""))

    # 1. consigne
    if parent is not None:
        try:
            definition_parent = _concept_parent(parent)
        except (OSError, ValueError) as ex:
            return _abandon(carnet, numero, e, f"concept de départ {parent.fichier or parent.id} illisible : {ex}",
                            dossier, trace)
        consigne = (consigne_tester(carnet, e, definition_parent) if e
                    else consigne_ameliorer(carnet, parent, definition_parent))
    else:
        consigne = consigne_explorer(carnet)
    lecons = carnet.pour_prompt(compact=reglages.moteur == "ouvert", parent=parent.id if parent else None)

    # 2. conception (sans calcul rapide pendant une expérience)
    session = concepteur(reglages, consigne, lecons, [] if parent else None, operation == "explorer",
                         d / "concepteur", afficher)
    if not session.concepts:
        return _abandon(carnet, numero, e, f"l'IA n'a pas enregistré de concept ({session.arret})", dossier, trace)
    chemin = Path(session.concepts[-1])
    definition = json.loads(chemin.read_text(encoding="utf-8"))
    trace["concept"] = _relatif(chemin)

    # 3. prédiction enregistrée avant tout calcul
    source = f"concepteur:{reglages.nom_modele()}"
    if operation == "ameliorer":
        e = enregistrer_prediction(carnet, definition, parent.id, source)
        trace["experience"] = e.id
    pr = enregistrer_prediction_physique(carnet, definition, chemin, source, e.id if e else None)
    if pr is not None:
        trace["prediction"] = pr.id
    carnet.noter(f"cycle {numero} ({operation}) : concept {_relatif(chemin)}" + (f", expérience {e.id}" if e else ""))
    carnet.enregistrer()
    _journal(dossier, f"Concept : `{_relatif(chemin)}`, {definition.get('nom', '')}."
             + (f"\n\nPrédiction enregistrée avant le calcul : {e.texte()}" if e else "")
             + (f"\n\nPrédiction physique enregistrée avant le calcul : {pr.texte()}" if pr else ""))

    # 4. mini-optimisation, depuis les cotes du parent pour un enfant
    opt = optimiseur(chemin, reglages, [parent.valeurs] if parent else None, d / "optimisation", afficher)
    trace["optimisation"] = {k: opt.get(k) for k in ("essais", "faisables", "kg_par_kN", "faisable")}
    info = {k: opt[k] for k in ("essais", "faisables", "budget", "taille", "graine", "duree_s", "reprise") if k in opt}
    if parent is not None:
        info["depart"] = parent.id
    lien = {"origine": f"ia:{reglages.nom_modele()}", "parent": parent.id if parent else None,
            "experience": e.id if e else None, "optimisation": info}

    # 5. fiche à la maille des fiches
    if opt.get("valeurs") is None:
        from gabarit.analyse.compte_rendu import CompteRendu

        cr = CompteRendu(definition.get("nom", chemin.stem), {}, reglages.taille_fiche, False,
                         ["aucun essai de l'optimisation n'a pu être calculé"])
        fiche = carnet.ajouter_fiche(cr, None, fichier=chemin, **lien)
    else:
        conv = {"taille": opt["taille"], "kg_par_kN": opt["kg_par_kN"]} if opt.get("kg_par_kN") else {}
        fiche = mesureur(carnet, chemin, opt["valeurs"], reglages, d / "fiche", convergence=conv, **lien)
    trace["fiche"] = fiche.id
    _journal(dossier, f"Optimisation : {opt.get('essais')} essais, {opt.get('faisables')} faisables, meilleur "
             f"{_nb(opt.get('kg_par_kN'), 4)} kg/kN à {_nb(opt.get('taille'), 0)} mm.\n\nFiche : {fiche.resume()}")

    # 6. verdict
    if e is not None:
        carnet.conclure_experience(e.id, fiche.id)
        trace["verdict"] = e.verdict
        _journal(dossier, f"**Verdict** : {e.texte()}")
        afficher(f"\n[verdict] {e.texte()}\n")
    if pr is not None:
        carnet.conclure_prediction(pr.id, fiche.id)
        trace["verdict_prediction"] = pr.verdict or pr.statut
        _journal(dossier, f"**Prédiction physique** : {pr.texte()}")
        afficher(f"\n[prédiction] {pr.texte()}\n")
    carnet.cycles = numero
    carnet.enregistrer()

    # 7. analyse
    place = max(0, min(EXPERIENCES_MAX, ATTENTE_MAX - len(carnet.experiences_a_faire())))
    bilan = analyste(reglages, carnet, [fiche.id], place, afficher)
    if bilan is not None:
        trace["analyse"] = {"changements": len(bilan.changements), "rejets": len(bilan.rejets)}
        carnet.enregistrer()
        _journal(dossier, f"**Analyse** ({reglages.nom_modele_analyste()}) :\n\n{bilan.texte()}")
    return trace


def _abandon(carnet: Carnet, numero: int, e: Experience | None, raison: str, dossier: Path, trace: dict) -> dict:
    if e is not None:
        carnet.conclure_experience(e.id, None, note=raison)
    carnet.noter(f"cycle {numero} sans concept : {raison}")
    carnet.cycles = numero
    carnet.enregistrer()
    _journal(dossier, f"Cycle sans concept : {raison}.")
    trace["abandon"] = raison
    return trace


def _erreur_ia(ex: Exception) -> bool:
    """Panne de l'IA (réseau, serveur, quota, clé) : le cycle est perdu, la boucle peut continuer. Une autre
    erreur (CalculiX absent, bug) arrête la boucle : la répéter ne servirait à rien."""
    import urllib.error

    if isinstance(ex, (urllib.error.URLError, ConnectionError, TimeoutError)):  # serveur arrêté, délai dépassé
        return True
    from gabarit.ia.ouvert import ErreurServeur

    if isinstance(ex, ErreurServeur):
        return True
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(ex, anthropic.APIError)


def _cle_refusee(ex: Exception) -> bool:
    code = getattr(ex, "code", None) or getattr(ex, "status_code", None)
    return code in (401, 403)


def _journal(dossier: Path, texte: str) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    with (dossier / "journal.md").open("a", encoding="utf-8") as f:
        f.write(texte.rstrip() + "\n\n")


def _relatif(chemin: Path) -> str:
    p = Path(chemin).resolve()
    return str(p.relative_to(RACINE)) if p.is_relative_to(RACINE) else str(p)


def _nb(x, n: int) -> str:
    return f"{x:.{n}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "-"
