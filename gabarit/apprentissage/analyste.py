"""IA Analyste : explique pourquoi un concept fait mieux ou moins bien qu'un autre en kg/kN, en tire des leçons
et propose des expériences pour les vérifier. Claude (SDK anthropic) ou un modèle ouvert (API compatible OpenAI).

L'IA reçoit les fiches du carnet (mesures de Gabarit, décomposition kg/kN = 1000 rho/Re x L_eq / eta, bilan de
chaque face, point critique, convergence au maillage) et les leçons existantes. Elle répond en JSON :
- une explication des écarts ;
- des leçons (nouvelles, précisées ou appuyées), chacune avec des comparaisons entre fiches : l'IA nomme deux
  fiches, une grandeur et le sens attendu, Gabarit lit les valeurs et tranche. Une leçon nouvelle dont toutes
  les comparaisons vont contre les mesures est refusée ;
- des expériences : modifier UNE chose d'un concept parent, avec la grandeur qui doit bouger et dans quel sens.
  La prédiction est enregistrée avant le calcul ; c'est la boucle d'apprentissage qui fera l'expérience.
Rien de ce que l'IA écrit n'est pris pour une mesure.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from gabarit.apprentissage.carnet import (
    GRANDEURS,
    METRIQUES_FACE,
    SEUIL_DEFAUT_PCT,
    Carnet,
    Fiche,
    grandeur_valide,
    texte_preuve,
    valeur,
)

MODELE = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 32000
EXPERIENCES_MAX = 3
LECONS_MAX = 6

SCHEMA = {
    "type": "object",
    "properties": {
        "explication": {"type": "string"},
        "lecons": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "action": {"type": "string", "enum": ["nouvelle", "preciser", "appuyer"]},
                "enonce": {"type": "string"},
                "mecanisme": {"type": "string"},
                "portee": {"type": "string"},
                "comparaisons": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"fiche_a": {"type": "string"}, "fiche_b": {"type": "string"},
                                   "grandeur": {"type": "string"}, "attendu": {"type": "string", "enum": ["<", ">"]}},
                    "required": ["fiche_a", "fiche_b", "grandeur", "attendu"],
                    "additionalProperties": False}},
            },
            "required": ["id", "action", "enonce", "mecanisme", "portee", "comparaisons"],
            "additionalProperties": False}},
        "experiences": {"type": "array", "items": {
            "type": "object",
            "properties": {"lecon": {"type": "string"}, "parent": {"type": "string"},
                           "modification": {"type": "string"}, "grandeur": {"type": "string"},
                           "sens": {"type": "string", "enum": ["baisse", "hausse"]}},
            "required": ["lecon", "parent", "modification", "grandeur", "sens"],
            "additionalProperties": False}},
    },
    "required": ["explication", "lecons", "experiences"],
    "additionalProperties": False,
}


@dataclass
class BilanAnalyse:
    explication: str = ""
    changements: list[str] = field(default_factory=list)
    rejets: list[str] = field(default_factory=list)
    jetons: dict = field(default_factory=dict)

    def texte(self) -> str:
        t = [f"Explication : {self.explication}"] if self.explication else []
        t += [f"+ {c}" for c in self.changements] + [f"- refusé : {r}" for r in self.rejets]
        return "\n".join(t)


def prompt_systeme(compact: bool = False) -> str:
    grandeurs = "; ".join(f"{k} ({v[0]}, mieux en {v[1]})" for k, v in GRANDEURS.items())
    metriques = ", ".join(METRIQUES_FACE)
    texte = f"""Tu es l'IA Analyste de Gabarit, un outil qui cherche la pièce mécano-soudée en tôle la plus \
performante (le moins de kg par kN tenu) pour un cahier des charges. Des concepts ont été optimisés puis \
calculés par éléments finis ; tu reçois leurs fiches. Ton travail : comprendre POURQUOI un concept tient plus \
de kN par kg qu'un autre, et en tirer des leçons générales, utiles pour concevoir les suivants.

## Ce que mesurent les fiches
Identité exacte : kg/kN = 1000 rho / Re x L_eq / eta.
- L_eq (mm), longueur équivalente du chemin d'effort = somme sur la matière de V x contrainte moyenne / F. \
Petite quand l'effort va droit aux fixations en traction ou compression ; grande avec des détours ou de la flexion.
- eta, utilisation de la matière = contrainte moyenne / contrainte au point critique. 1 si toute la matière \
travaillait autant que le point le plus chargé ; une tôle en flexion pure plafonne à 0,5 ; la matière peu \
chargée la fait baisser.
- Le rapport des kg/kN de deux concepts est exactement le rapport des L_eq divisé par celui des eta, et aussi \
le rapport des masses multiplié par celui des contraintes utiles.
- Chaque face a sa part de masse, d'énergie, de L_eq, son eta, sa part de flexion et la part de sa surface \
sous 20 % de la contrainte utile (matière peu utile).
- La contrainte utile exclut les points singuliers (vis, appui, bouts de jonction) ; « convergence » donne le \
kg/kN à une autre maille : un écart de quelques % est du bruit de calcul, pas une différence de conception.

## Ce que tu rends (JSON imposé)
- explication : pourquoi les concepts nouveaux font mieux ou moins bien que leurs références, en citant les \
fiches par leur numéro (F001...) et la décomposition (L_eq, eta, faces, point critique, limite).
- lecons : au plus {LECONS_MAX}. Une leçon est une règle de conception générale et vérifiable (pas une \
description d'un concept) : énoncé, mécanisme physique (chemin d'effort, flexion ou membrane, flambement, \
concentration de contrainte...), portée (dans quels cas elle vaut). action « nouvelle » (id vide), \
« preciser » (id d'une leçon existante dont tu réécris l'énoncé, le mécanisme ou la portée) ou « appuyer » \
(id existant, tu ajoutes seulement des comparaisons). comparaisons : deux fiches (fiche_a, fiche_b), une \
grandeur et le sens attendu (« < » : fiche_a a la plus petite valeur). N'écris AUCUN nombre dans les \
comparaisons : Gabarit lit les valeurs dans les fiches et vérifie le sens ; une leçon nouvelle que toutes ses \
comparaisons contredisent est refusée.
- experiences : au plus {EXPERIENCES_MAX}. Une expérience modifie UNE seule chose d'un concept parent (fiche) \
pour tester une leçon (id existant, « N1 », « N2 »... pour la 1re, 2e leçon nouvelle de ta réponse, ou vide) et \
prédit le sens de variation d'une grandeur. Le seuil de décision est {SEUIL_DEFAUT_PCT:g} % : choisis une \
modification assez forte pour que l'effet le dépasse.
Grandeurs possibles : {grandeurs} ; ou « face:<id de face>:<métrique> » avec métrique parmi {metriques}.

## Règles
- Appuie-toi sur les chiffres des fiches, pas sur des idées reçues ; si les données ne permettent pas de \
conclure, dis-le dans l'explication et propose plutôt une expérience.
- Deux concepts diffèrent par plusieurs choses : une comparaison montre une corrélation, pas une cause. \
C'est l'expérience (une seule modification) qui établit la cause.
- Ne répète pas une leçon existante : précise-la ou appuie-la.
- Les leçons « contredites » par l'expérience sont des résultats, pas des erreurs : ne les reformule pas pour \
les sauver, propose plutôt la règle qui explique les deux résultats."""
    if compact:
        texte += "\nSois bref : trois leçons et deux expériences au plus."
    return texte


def prompt_demande(carnet: Carnet, nouvelles: list[str], compact: bool = False,
                   experiences_max: int = EXPERIENCES_MAX) -> str:
    """Les fiches à analyser, leurs références (parent, meilleurs concepts), les statistiques et les leçons
    existantes."""
    ids = list(dict.fromkeys(nouvelles))
    for i in list(ids):
        f = carnet.fiche(i)
        if f.parent and carnet.a_fiche(f.parent):
            ids.append(f.parent)
    ids += [f.id for f in carnet.meilleures(2 if compact else 3)]
    ids = list(dict.fromkeys(ids))
    lignes = [f"## Fiches à analyser : {', '.join(nouvelles)}", ""]
    for i in ids:
        lignes += [texte_fiche(carnet.fiche(i), compact), ""]
    paires = []
    for i in nouvelles:
        f = carnet.fiche(i)
        refs = [f.parent] if f.parent and carnet.a_fiche(f.parent) else []
        refs += [m.id for m in carnet.meilleures(1) if m.id != i]
        for r in dict.fromkeys(refs):
            paires.append(comparer_fiches(f, carnet.fiche(r)))
    if paires:
        lignes += ["## Comparaisons calculées par Gabarit"] + [f"- {p}" for p in paires] + [""]
    stats = carnet.texte_statistiques()
    if stats and not compact:
        lignes += ["## Statistiques sur toutes les fiches (Gabarit)"] + [f"- {t}" for t in stats] + [""]
    for i in nouvelles:
        f = carnet.fiche(i)
        if f.experience and any(e.id == f.experience for e in carnet.experiences):
            lignes.append(f"## Expérience à laquelle répond {i}\n{carnet.experience(f.experience).texte()}\n")
    if carnet.lecons:
        lignes.append("## Leçons existantes")
        for x in carnet.lecons:
            lignes.append(f"- {x.id} ({x.statut}) : {x.enonce}" + ("" if compact else f" Mécanisme : {x.mecanisme} "
                                                                      f"Portée : {x.portee} Preuves : {x.bilan_preuves()}."))
            if not compact:
                lignes += [f"  - {texte_preuve(p)}" for p in x.preuves[-4:]]
    else:
        lignes.append("## Leçons existantes\nAucune pour l'instant.")
    a_faire = carnet.experiences_a_faire()
    if a_faire:
        lignes.append("\n## Expériences déjà prévues (ne pas les reproposer)")
        lignes += [f"- {e.texte()}" for e in a_faire]
    faites = [e for e in carnet.experiences if e.statut != "a_faire"]
    if faites and not compact:
        lignes.append("\n## Expériences faites")
        lignes += [f"- {e.texte()}" for e in faites[-8:]]
    lignes.append("\n" + (f"Propose au plus {experiences_max} expérience(s)." if experiences_max > 0 else
                          "Ne propose pas d'expérience nouvelle : assez sont déjà prévues (liste « experiences » vide)."))
    lignes.append("Analyse ces fiches et réponds dans le format JSON demandé.")
    return "\n".join(lignes)


def texte_fiche(f: Fiche, compact: bool = False) -> str:
    """Une fiche du carnet en texte, pour une IA."""
    lignes = [f"### {f.resume()}"]
    if f.parent:
        lignes.append(f"Dérive de {f.parent}" + (f" (expérience {f.experience})" if f.experience else "") + ".")
    if f.principe and not compact:
        lignes.append(f"Principe : {f.principe}")
    if f.caracteristiques:
        lignes.append("Caractéristiques : " + ", ".join(f"{k} {v}" for k, v in f.caracteristiques.items()
                                                        if isinstance(v, (int, float)) and v is not None))
    if f.point_critique.get("texte"):
        lignes.append(f"Point critique : {f.point_critique['texte']}")
    if f.marges:
        lignes.append("Marges (> 1 : limite respectée) : " + ", ".join(f"{k} {_nb(v, 3)}" for k, v in f.marges.items()))
    o = f.optimisation
    if o.get("essais"):
        lignes.append(f"Optimisation : {o['essais']} essais ({o.get('faisables', '?')} faisables), maille {_nb(o.get('taille'), 0)} mm"
                      + (f", partie des cotes de {o['depart']}" if o.get("depart") else "") + ".")
    k, kc = f.mesures.get("kg_par_kN"), f.convergence.get("kg_par_kN")
    if k and kc:
        lignes.append(f"Convergence : {kc:.4f} kg/kN à {f.convergence.get('taille', '?')} mm contre {k:.4f} à "
                      f"{f.taille:g} mm ({100 * (kc / k - 1):+.1f} %).")
    if f.faces:
        lignes.append("Faces (masse, énergie, L_eq, eta, flexion, surface peu utile) :")
        for x in sorted(f.faces, key=lambda x: -(x.get("part_energie") or 0))[: (5 if compact else 12)]:
            lignes.append(f"- {x['id']} (t {x['t']:g}) : {_pc(x.get('part_masse'))}, {_pc(x.get('part_energie'))}, "
                          f"{_nb(x.get('l_eq_mm'), 0)} mm, {_nb(x.get('eta'), 2)}, {_pc(x.get('part_flexion'))}, "
                          f"{_pc(x.get('part_peu_utile'))}")
    if f.pistes and not compact:
        lignes += [f"Piste (règle automatique) : {p}" for p in f.pistes[:3]]
    if f.erreurs and not f.mesures:
        lignes.append("Erreurs : " + "; ".join(f.erreurs[:3]))
    return "\n".join(lignes)


def comparer_fiches(a: Fiche, b: Fiche) -> str:
    """Décomposition exacte du rapport des kg/kN de a sur b (mêmes matériau et effort)."""
    ka, kb = valeur(a, "kg_par_kN"), valeur(b, "kg_par_kN")
    la, lb = valeur(a, "l_eq_mm"), valeur(b, "l_eq_mm")
    ea, eb = valeur(a, "eta"), valeur(b, "eta")
    ma, mb = valeur(a, "masse_kg"), valeur(b, "masse_kg")
    sa, sb = valeur(a, "sigma_utile"), valeur(b, "sigma_utile")
    if not all(math.isfinite(x) and x > 0 for x in (ka, kb, la, lb, ea, eb, ma, mb, sa, sb)):
        return f"{a.id} / {b.id} : comparaison impossible (mesures manquantes)."
    t = (f"{a.id} / {b.id} : kg/kN x{ka / kb:.3f} = L_eq x{la / lb:.3f} / eta x{ea / eb:.3f} "
         f"= masse x{ma / mb:.3f} x contrainte utile x{sa / sb:.3f}.")
    fa, fb = {x["id"]: x for x in a.faces}, {x["id"]: x for x in b.faces}
    ecarts = sorted(((fa.get(i, {}).get("l_eq_mm") or 0) - (fb.get(i, {}).get("l_eq_mm") or 0), i) for i in set(fa) | set(fb))
    ecarts.sort(key=lambda e: -abs(e[0]))
    if ecarts:
        t += " Faces qui pèsent le plus dans l'écart de L_eq : " + "; ".join(
            f"{i} {_nb(fa.get(i, {}).get('l_eq_mm'), 0)} contre {_nb(fb.get(i, {}).get('l_eq_mm'), 0)} mm"
            for _, i in ecarts[:3]) + "."
    return t


def appliquer(carnet: Carnet, sortie: dict, source: str = "analyste", experiences_max: int = EXPERIENCES_MAX) -> BilanAnalyse:
    """Range la réponse de l'IA dans le carnet après contrôle : fiches et leçons existantes, grandeurs connues,
    comparaisons vérifiées sur les mesures. Renvoie ce qui a été fait et ce qui a été refusé (et pourquoi)."""
    bilan = BilanAnalyse(str(sortie.get("explication", "")).strip())
    nouvelles: dict[str, str] = {}  # « N1 » -> id attribué
    lecons = list(sortie.get("lecons") or [])
    if len(lecons) > LECONS_MAX:
        bilan.rejets.append(f"{len(lecons) - LECONS_MAX} leçon(s) au-delà de la limite de {LECONS_MAX}")
    for k, item in enumerate(lecons[:LECONS_MAX], start=1):
        if not isinstance(item, dict):
            bilan.rejets.append(f"leçon {k} illisible")
            continue
        action = item.get("action")
        comps, refus = _comparaisons(carnet, item.get("comparaisons") or [])
        bilan.rejets += refus
        if action == "nouvelle":
            enonce = str(item.get("enonce", "")).strip()
            if not enonce:
                bilan.rejets.append(f"leçon nouvelle {k} sans énoncé")
                continue
            mesures = [_verdict_comparaison(carnet, c) for c in comps]
            if comps and all(v == "contraire" for v in mesures):
                bilan.rejets.append(f"leçon « {enonce[:80]} » : toutes ses comparaisons vont contre les mesures")
                continue
            x = carnet.ajouter_lecon(enonce, str(item.get("mecanisme", "")), str(item.get("portee", "")), source)
            for c in comps:
                carnet.preuve_comparaison(x, *c)
            carnet.mettre_a_jour_statut(x)
            nouvelles[f"N{k}"] = x.id
            bilan.changements.append(f"leçon {x.id} ({x.statut}) : {x.enonce} [{x.bilan_preuves()}]")
        elif action in ("preciser", "appuyer"):
            ident = str(item.get("id", "")).strip()
            if not any(x.id == ident for x in carnet.lecons):
                bilan.rejets.append(f"leçon {ident or '?'} inconnue ({action})")
                continue
            x = carnet.lecon(ident)
            if action == "preciser":
                for cle in ("enonce", "mecanisme", "portee"):
                    v = str(item.get(cle, "")).strip()
                    if v:
                        setattr(x, cle, v)
            for c in comps:
                carnet.preuve_comparaison(x, *c)
            carnet.mettre_a_jour_statut(x)
            bilan.changements.append(f"leçon {x.id} {'précisée' if action == 'preciser' else 'appuyée'} ({x.statut}) : "
                                     f"{x.enonce} [{x.bilan_preuves()}]")
        else:
            bilan.rejets.append(f"leçon {k} : action inconnue {action!r}")
    proposees = list(sortie.get("experiences") or [])
    if len(proposees) > max(0, experiences_max):
        bilan.rejets.append(f"{len(proposees) - max(0, experiences_max)} expérience(s) au-delà de la limite de {max(0, experiences_max)}")
    for k, item in enumerate(proposees[:max(0, experiences_max)], start=1):
        if not isinstance(item, dict):
            bilan.rejets.append(f"expérience {k} illisible")
            continue
        lecon = str(item.get("lecon", "")).strip()
        lecon = nouvelles.get(lecon, lecon) or None
        if lecon and not any(x.id == lecon for x in carnet.lecons):
            bilan.rejets.append(f"expérience {k} : leçon {lecon} inconnue, gardée sans leçon")
            lecon = None
        try:
            e = carnet.ajouter_experience(str(item.get("parent", "")).strip(), str(item.get("modification", "")),
                                          str(item.get("grandeur", "kg_par_kN")).strip(), str(item.get("sens", "")),
                                          lecon, source=source)
        except ValueError as ex:
            bilan.rejets.append(f"expérience {k} : {ex}")
            continue
        if not e.modification:
            carnet.experiences.remove(e)
            bilan.rejets.append(f"expérience {k} : pas de modification décrite")
            continue
        bilan.changements.append(f"expérience prévue {e.texte()}")
    if bilan.explication:
        carnet.noter(f"analyse ({source}) : {bilan.explication[:500]}")
    return bilan


def _comparaisons(carnet: Carnet, items: list) -> tuple[list[tuple], list[str]]:
    ok, refus = [], []
    for c in items:
        if not isinstance(c, dict):
            refus.append("comparaison illisible")
            continue
        a, b = str(c.get("fiche_a", "")).strip(), str(c.get("fiche_b", "")).strip()
        g, att = str(c.get("grandeur", "")).strip(), str(c.get("attendu", "")).strip()
        if not (carnet.a_fiche(a) and carnet.a_fiche(b)) or a == b:
            refus.append(f"comparaison {a} / {b} : fiche inconnue")
        elif not grandeur_valide(g):
            refus.append(f"comparaison {a} / {b} : grandeur inconnue {g!r}")
        elif att not in ("<", ">"):
            refus.append(f"comparaison {a} / {b} : sens {att!r} (attendu < ou >)")
        elif not (math.isfinite(valeur(carnet.fiche(a), g)) and math.isfinite(valeur(carnet.fiche(b), g))):
            refus.append(f"comparaison {a} / {b} : {g} non mesuré")
        else:
            ok.append((a, b, g, att))
    return ok, refus


def _verdict_comparaison(carnet: Carnet, c: tuple) -> str:
    a, b, g, att = c
    va, vb = valeur(carnet.fiche(a), g), valeur(carnet.fiche(b), g)
    if abs(va - vb) <= SEUIL_DEFAUT_PCT / 100 * abs(vb):
        return "non_concluante"
    return "conforme" if (va < vb) == (att == "<") else "contraire"


# ---------- appels aux modèles ----------
def analyser_claude(carnet: Carnet, nouvelles: list[str], modele: str = MODELE, effort: str = EFFORT,
                    client=None, experiences_max: int = EXPERIENCES_MAX, afficher=print) -> BilanAnalyse:
    """Analyse par Claude, réponse en JSON imposé (sortie structurée). Clé : ANTHROPIC_API_KEY."""
    import anthropic

    client = client or anthropic.Anthropic(max_retries=4)
    with client.messages.stream(
        model=modele, max_tokens=MAX_TOKENS, system=prompt_systeme(),
        messages=[{"role": "user", "content": prompt_demande(carnet, nouvelles, experiences_max=experiences_max)}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
    ) as flux:
        reponse = flux.get_final_message()
    if reponse.stop_reason == "refusal":
        return BilanAnalyse(rejets=["le modèle a refusé de répondre"])
    if reponse.stop_reason == "max_tokens":
        return BilanAnalyse(rejets=[f"réponse coupée à {MAX_TOKENS} jetons"])
    texte = next((b.text for b in reponse.content if b.type == "text"), "")
    sortie = lire_json(texte)
    if sortie is None:
        return BilanAnalyse(rejets=["réponse illisible (JSON attendu)"])
    bilan = appliquer(carnet, sortie, f"analyste:{modele}", experiences_max)
    u = reponse.usage
    bilan.jetons = {"entree": u.input_tokens, "sortie": u.output_tokens}
    afficher(bilan.texte())
    return bilan


def analyser_ouvert(carnet: Carnet, nouvelles: list[str], url: str, modele: str, cle: str = "",
                    delai: float = 1800.0, compact: bool = True, experiences_max: int = EXPERIENCES_MAX,
                    afficher=print) -> BilanAnalyse:
    """Analyse par un modèle ouvert (Ollama, LM Studio...). Le schéma JSON est demandé au serveur
    (response_format) ; s'il le refuse, la réponse est lue dans le texte."""
    from gabarit.ia.ouvert import ErreurServeur, _poster

    entetes = {"Content-Type": "application/json", **({"Authorization": f"Bearer {cle}"} if cle else {})}
    messages = [{"role": "system", "content": prompt_systeme(compact) + "\nRéponds uniquement par l'objet JSON."},
                {"role": "user", "content": prompt_demande(carnet, nouvelles, compact, experiences_max)}]
    corps = {"model": modele, "messages": messages,
             "response_format": {"type": "json_schema", "json_schema": {"name": "analyse", "schema": SCHEMA}}}
    try:
        donnees = _poster(f"{url.rstrip('/')}/chat/completions", corps, entetes, delai)
    except ErreurServeur as e:
        if e.code not in (400, 422):
            raise
        corps.pop("response_format")  # serveur sans sortie structurée : JSON lu dans le texte
        donnees = _poster(f"{url.rstrip('/')}/chat/completions", corps, entetes, delai)
    texte = (donnees["choices"][0]["message"].get("content") or "")
    sortie = lire_json(re.sub(r"<think>.*?</think>", "", texte, flags=re.DOTALL))
    if sortie is None:
        return BilanAnalyse(rejets=["réponse illisible (JSON attendu)"])
    bilan = appliquer(carnet, sortie, f"analyste:{modele}", experiences_max)
    u = donnees.get("usage") or {}
    bilan.jetons = {"entree": u.get("prompt_tokens", 0), "sortie": u.get("completion_tokens", 0)}
    afficher(bilan.texte())
    return bilan


def lire_json(texte: str) -> dict | None:
    """L'objet JSON d'une réponse : texte entier, sinon le dernier bloc ```json, sinon le plus grand {...}."""
    texte = texte.strip()
    candidats = [texte] + re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", texte, flags=re.DOTALL)[::-1]
    if "{" in texte:
        candidats.append(texte[texte.index("{"): texte.rindex("}") + 1])
    for c in candidats:
        try:
            d = json.loads(c)
        except ValueError:
            continue
        if isinstance(d, dict):
            return d
    return None


def _pc(x) -> str:
    return f"{100 * x:.0f} %" if isinstance(x, (int, float)) and math.isfinite(x) else "-"


def _nb(x, n: int) -> str:
    return f"{x:.{n}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "-"
