"""IA Concepteur (phase 4) : Claude propose des formes générales au format déclaratif et les corrige lui-même.

Déroulé : le cahier des charges, le format des concepts, les règles de fabrication, ce que l'on a appris
des concepts précédents et un concept d'exemple partent dans le prompt système. Claude écrit un concept,
appelle l'outil `verifier_concept` (géométrie au point central et sur des tirages aléatoires, calcul
rapide), corrige d'après le rapport, puis appelle `enregistrer_concept`. Rien n'est enregistré sans
passer les contrôles. Les concepts acceptés vont dans concepts/ia/ et s'optimisent ensuite avec
exemples/optimiser_concept.py.

Clé API : variable d'environnement ANTHROPIC_API_KEY (lue par le SDK anthropic).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts import format as format_concepts
from gabarit.ia.outils import PART_TIRAGES_VALIDES_MIN, enregistrer, examiner_concept

MODELE = "claude-opus-5-5"
EFFORT = "high"  # low | medium | high | xhigh | max ; la conception est un travail de raisonnement
MAX_TOKENS = 64000  # un tour peut contenir plusieurs concepts complets de quelques milliers de jetons chacun
MAX_TOKENS_PLAFOND = 128000  # sortie maximale du modèle : une réponse coupée est relancée jusque-là
TOURS_MAX = 40
RACINE = Path(__file__).resolve().parents[2]

_SCHEMA_CONCEPT = {
    "type": "object",
    "properties": {"concept": {"type": "object", "description": "Le concept complet, au format décrit dans le prompt système."}},
    "required": ["concept"],
}
OUTILS = [
    {
        "name": "verifier_concept",
        "description": (
            "Contrôle un concept sans l'enregistrer. Construit la géométrie au point central des variables et "
            "sur des tirages aléatoires, liste les erreurs (structure du fichier, encombrement, plis, aile "
            "minimale, soudures qui ne se touchent pas, trous de fixation utilisables, appui), puis lance un "
            "calcul statique grossier au point central (masse, effort tenu avant plastification, zone la plus "
            "chargée, traction dans les vis). Prend 10 à 40 s. À appeler après chaque modification."),
        "input_schema": _SCHEMA_CONCEPT,
        "eager_input_streaming": True,
    },
    {
        "name": "enregistrer_concept",
        "description": (
            "Enregistre un concept définitif pour l'optimisation. Refusé tant que le concept n'est pas valide "
            f"au point central et sur au moins {PART_TIRAGES_VALIDES_MIN:.0%} des tirages aléatoires. "
            "Renvoie le chemin du fichier."),
        "input_schema": _SCHEMA_CONCEPT,
        "eager_input_streaming": True,
    },
]


def prompt_systeme(cdc: CahierDesCharges, exemples: list[Path]) -> str:
    ex = "\n\n".join(f"Exemple ({p.name}) :\n```json\n{p.read_text(encoding='utf-8').strip()}\n```" for p in exemples)
    return f"""Tu es l'IA Concepteur de Gabarit, un outil qui cherche la pièce la plus performante pour un cahier \
des charges. Ton rôle : proposer des formes générales de pièces mécano-soudées en tôle pliée, chacune décrite \
comme un concept paramétré. Un optimiseur (plan d'expériences, optimisation bayésienne, affinage) fera ensuite \
varier toutes les variables et évaluera chaque combinaison par éléments finis coques sur l'assemblage complet. \
Une autre IA analysera les résultats.

## Cahier des charges
```json
{cdc.model_dump_json(indent=None)}
```

## Repère et chargement
Repère du socle : X perpendiculaire à la face de fixation (face avant du socle en x = {format_concepts.X_FACE:g}, \
la pièce est du côté x > {format_concepts.X_FACE:g}), Y le long de cette face (nervures latérales en |y| = \
{cdc.encombrement.y_max:g}), Z vertical (dessus de la semelle en z = {cdc.encombrement.z_min:g}). La semelle \
s'arrête en x = {cdc.encombrement.x_max:g}. L'effort de {cdc.chargements[0].force:g} N vers le bas passe par \
l'adaptateur, un appui de diamètre 50 centré en x = {format_concepts.X_FACE + cdc.chargements[0].distance_encastrement:g}, \
y = 0, posé sur la face d'appui. La fixation se fait par vis M10 8.8 dans la grille de trous du socle ; le dos \
plaqué sur le socle peut aussi s'y appuyer (contact unilatéral, pris en compte dans le calcul).

## Ce que l'optimiseur minimise
Masse / effort tenu (kg/kN), sous contraintes : pas de plastification à {cdc.chargements[0].force:g} N \
(contrainte de Von Mises hors singularités au bord des vis et sous l'appui), facteur de flambement linéaire \
≥ 3, traction par vis ≤ résistance d'une M10 8.8 (33,4 kN).

## Format d'un concept
{format_concepts.__doc__}
Précisions :
- Les points des faces sont des listes [x, y, z] dont chaque coordonnée est un nombre ou une expression.
- Les variables sont évaluées avant les « derivees », qui sont évaluées dans l'ordre (une dérivée peut \
utiliser les précédentes). Les retours se construisent sur l'arête [i, j] de leur face, vers « direction ».
- Une variable « retour » vaut 0 (pli absent) ou une hauteur entre l'aile minimale de pliage de sa tôle et \
« max » : c'est l'optimiseur qui décide si le retour vaut sa masse. Une face optionnelle (« si ») sert pour \
les faces qui n'existent qu'avec un retour.
- La face de fixation doit être dans un plan x = constante (surface moyenne à {format_concepts.X_FACE:g} + \
épaisseur / 2) ; les vis utilisent les trous des colonnes y = ±colonnes_vis (ou y = 0 si 0) situés à au moins \
{format_concepts.MARGE_TROU:g} mm des bords de cette face. La face d'appui doit être horizontale et contenir \
le point (x = 320, y = 0).
- Contrôles appliqués : épaisseurs de l'atelier, nombre de tôles ≤ {cdc.procede.nombre_pieces_max}, rien à \
moins de {format_concepts.DEGAGEMENT_SEMELLE:g} mm de la semelle, rien au-delà de x = {cdc.encombrement.x_max:g} \
ni à |y| + t/2 > {cdc.encombrement.y_max:g}, rien derrière la face du socle, plis d'angle intérieur ≥ \
{format_concepts.ANGLE_PLI_MIN:g}°, ailes ≥ aile minimale (0,7 × vé, vé standard le plus proche de 8 × t), \
chaque soudure déclarée doit exister (arête commune d'au moins 10 mm, jonction en T ou doublage), assemblage \
d'un seul tenant.

## Bonnes pratiques
- Chaque cote est une variable ou une dérivée de variables : pas de valeur figée sans raison. Classe bien les \
variables (épaisseur, choix discret, réel borné, fraction, retour optionnel).
- Paramètre avec des fractions mises à l'échelle dans les dérivées pour que toute combinaison des variables \
donne une pièce fabricable (par exemple une hauteur de jambe exprimée entre son minimum et l'espace disponible) : \
l'outil de contrôle compte la part des tirages aléatoires valides, vise 80 % ou plus.
- Les justifications expliquent chaque choix de conception (tôle, face, pli, retour, soudure) : quel chemin \
d'effort, quelle rigidité, quel mode de ruine il vise.
- Propose des concepts qui diffèrent par leur principe (chemin d'effort, nombre de tôles, type de \
raidissement), pas seulement par leurs dimensions.

## Ce qu'on a appris sur ce cas
- Une simple équerre en U (dos, dessus, deux ailes) plafonne vers 4 à 5 kN : le dessus plie localement sous \
l'appui, loin de tout raidisseur.
- Une diagonale (tôle en V sous le dessus) qui soutient le dessus près de l'appui, avec des retours sur ses \
bords, fait passer le facteur de flambement de 0,9 à plus de 30 et divise la contrainte par trois.
- Les contraintes se concentrent aux extrémités des lignes de soudure en T et aux jonctions en un seul point : \
préfère des liaisons sur toute la largeur et des retours qui se rejoignent bord à bord.
- La bande du dos entre le pli et les vis du haut est fortement sollicitée (le dessus tire dessus) ; écarter \
les vis et doubler le dos aident.

{ex}

## Méthode
Pour chaque concept : écris-le, appelle verifier_concept, corrige d'après le rapport jusqu'à ce qu'il soit \
valide, puis appelle enregistrer_concept. Travaille au plus deux concepts à la fois : un concept complet fait \
plusieurs milliers de jetons, et une réponse trop longue est coupée. Le calcul rapide n'est qu'un ordre de \
grandeur au point central : l'optimiseur trouvera les bonnes dimensions, ton travail est de lui donner une forme et un espace de variables \
où une bonne solution existe. Quand tous les concepts demandés sont enregistrés, termine par un court bilan en \
français : pour chaque concept, son principe et pourquoi il pourrait gagner."""


@dataclass
class Session:
    """Trace d'une conception : concepts enregistrés, journal lisible, consommation."""

    dossier: Path
    concepts: list[Path] = field(default_factory=list)
    bilan: str = ""
    jetons: dict = field(default_factory=lambda: {"entree": 0, "sortie": 0, "cache_lu": 0, "cache_ecrit": 0})
    arret: str = ""

    def journal(self, texte: str) -> None:
        self.dossier.mkdir(parents=True, exist_ok=True)
        with (self.dossier / "journal.md").open("a", encoding="utf-8") as f:
            f.write(texte.rstrip() + "\n\n")


def concevoir(n_concepts: int = 3, chemin_cdc: str | Path = RACINE / "cahiers" / "sae51_equerre.json",
              exemples: list[Path] | None = None, sortie: Path = RACINE / "concepts" / "ia",
              dossier_journal: Path | None = None, modele: str = MODELE, effort: str = EFFORT,
              consigne: str = "", calcul: bool = True, client: anthropic.Anthropic | None = None,
              afficher=print) -> Session:
    cdc = CahierDesCharges.charger(chemin_cdc)
    exemples = exemples if exemples is not None else [RACINE / "concepts" / "sae51_L_V.json"]
    session = Session(dossier_journal or RACINE / "resultats" / "concepteur" / time.strftime("%Y%m%d_%H%M%S"))
    client = client or anthropic.Anthropic(max_retries=4)
    systeme = [{"type": "text", "text": prompt_systeme(cdc, exemples), "cache_control": {"type": "ephemeral"}}]
    quoi = "un concept" if n_concepts == 1 else f"{n_concepts} concepts différents"
    demande = (f"Propose {quoi} pour ce cahier des charges, chacun vérifié puis enregistré."
               + (f"\n\nConsigne de l'utilisateur : {consigne}" if consigne else ""))
    messages = [{"role": "user", "content": demande}]
    session.journal(f"# IA Concepteur\n\nModèle {modele}, effort {effort}.\n\n**Demande** : {demande}")
    echecs_json = 0
    limite = MAX_TOKENS

    for tour in range(TOURS_MAX):
        try:
            with client.beta.messages.stream(
                model=modele, max_tokens=limite, system=systeme, messages=messages, tools=OUTILS,
                thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": effort},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                cache_control={"type": "ephemeral"},  # le préfixe de la conversation se relit depuis le cache
            ) as flux:
                for ev in flux:
                    if ev.type == "text":
                        afficher(ev.text, end="", flush=True)
                reponse = flux.get_final_message()
            echecs_json = 0
        except ValueError:  # entrée d'outil illisible, coupée avant la fin : on redemande le tour
            echecs_json += 1
            if echecs_json > 2:
                raise
            continue
        _compter(session, reponse.usage)
        contenu = _contenu_a_renvoyer(reponse.content)
        _tracer(session, contenu)

        if reponse.stop_reason == "refusal":
            details = getattr(reponse, "stop_details", None)
            session.arret = f"refus du modèle : {getattr(details, 'explanation', '') or details}"
            session.journal(f"**Arrêt** : {session.arret}")
            break
        if reponse.stop_reason == "max_tokens":  # réponse coupée : une entrée d'outil tronquée se lit comme un objet partiel
            if limite >= MAX_TOKENS_PLAFOND:
                session.arret = f"réponse tronquée à {limite} jetons (max_tokens), même avec la limite la plus haute"
                session.journal(f"**Arrêt** : {session.arret}")
                break
            limite = min(2 * limite, MAX_TOKENS_PLAFOND)  # on rejoue le même tour, sans exécuter ses outils
            afficher(f"\n[réponse coupée : le même tour est relancé avec une limite de {limite} jetons]\n")
            session.journal(f"Réponse coupée (max_tokens) : le même tour est relancé avec une limite de {limite} jetons.")
            continue
        if reponse.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": contenu})
            continue
        appels = [b for b in contenu if b.type == "tool_use"]
        if not appels:
            session.bilan = "\n".join(b.text for b in contenu if b.type == "text").strip()
            session.arret = "terminé"
            break

        resultats = []
        for b in appels:
            texte, erreur = _executer_outil(b.name, b.input, cdc, sortie, session, calcul)
            afficher(f"\n[{b.name}] {texte[:300]}{'…' if len(texte) > 300 else ''}\n")
            session.journal(f"**Résultat de {b.name}**{' (erreur)' if erreur else ''} :\n```json\n{texte}\n```")
            resultats.append({"type": "tool_result", "tool_use_id": b.id, "content": texte, "is_error": erreur})
        messages.append({"role": "assistant", "content": contenu})
        messages.append({"role": "user", "content": resultats})
    else:
        session.arret = f"limite de {TOURS_MAX} tours atteinte"

    session.journal(f"## Bilan\n\n{session.bilan or '(aucun)'}\n\nArrêt : {session.arret}. "
                    f"Concepts : {', '.join(_relatif(p) for p in session.concepts) or 'aucun'}. "
                    f"Jetons : {session.jetons}.")
    return session


def _executer_outil(nom: str, entree, cdc, sortie: Path, session: Session, calcul: bool) -> tuple[str, bool]:
    concept = entree.get("concept") if isinstance(entree, dict) else None
    if not isinstance(concept, dict):  # entrée tronquée ou mal formée : le parseur tolérant l'a laissée passer
        return json.dumps({"INVALID_JSON": "l'entrée doit être {\"concept\": {...}}"}, ensure_ascii=False), True
    if nom not in ("verifier_concept", "enregistrer_concept"):
        return f"outil inconnu : {nom}", True
    try:
        rapport = examiner_concept(concept, cdc, calcul=calcul and nom == "verifier_concept")
    except Exception as ex:  # un concept peut casser la géométrie de façon imprévue : on le dit à l'IA
        return json.dumps({"valide": False, "erreur_interne": str(ex)[:500]}, ensure_ascii=False), True
    if nom == "verifier_concept":
        return json.dumps(rapport, ensure_ascii=False), False
    if not rapport["valide"]:
        return json.dumps({"refuse": "le concept ne passe pas les contrôles", "rapport": rapport},
                          ensure_ascii=False), True
    chemin = enregistrer(concept, sortie)
    session.concepts.append(chemin)
    return json.dumps({"enregistre": _relatif(chemin), "rapport": rapport}, ensure_ascii=False), False


def _relatif(chemin: Path) -> str:
    return str(chemin.relative_to(RACINE)) if chemin.is_relative_to(RACINE) else str(chemin)


def _contenu_a_renvoyer(contenu: list) -> list:
    """Après un relais vers le modèle de secours en cours de réponse, les blocs de raisonnement et d'appel
    d'outil émis avant le dernier bloc « fallback » ne se renvoient pas ; le texte, si."""
    dernier = max((i for i, b in enumerate(contenu) if b.type == "fallback"), default=-1)
    garde = [b for b in contenu[:dernier] if b.type == "text"] if dernier >= 0 else []
    return garde + list(contenu[dernier + 1:])


def _compter(session: Session, usage) -> None:
    for cle, attr in (("entree", "input_tokens"), ("sortie", "output_tokens"),
                      ("cache_lu", "cache_read_input_tokens"), ("cache_ecrit", "cache_creation_input_tokens")):
        session.jetons[cle] += getattr(usage, attr, 0) or 0


def _tracer(session: Session, contenu: list) -> None:
    for b in contenu:
        if b.type == "thinking" and getattr(b, "thinking", ""):
            session.journal(f"> *Raisonnement (résumé)* : {b.thinking}")
        elif b.type == "text" and b.text.strip():
            session.journal(b.text)
        elif b.type == "tool_use":
            session.journal(f"**Appel {b.name}** :\n```json\n{json.dumps(b.input, indent=1, ensure_ascii=False)}\n```")
