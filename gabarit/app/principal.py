"""Interface de Gabarit (application locale NiceGUI) : cahier des charges, concepts, optimisation, résultats,
apprentissage.

Lancer depuis la racine du dépôt :  python -m gabarit.app   (ajouter --fenetre pour une fenêtre native)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from nicegui import app, run, ui

from gabarit.app.taches import RACINE, TACHES
from gabarit.app.vue3d import (
    faces_de_synthese,
    faces_du_concept,
    figure_essais,
    figure_faces,
    figure_grille,
    lire_essais,
)
from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.concepts.format import ConceptJSON

CONFIG = Path.home() / ".gabarit" / "config.json"
BUDGETS = {"Rapide (16 + 32 + 16 essais)": "16,32,16", "Normal (32 + 96 + 40 essais)": "32,96,40",
           "Poussé (64 + 192 + 60 essais)": "64,192,60"}
ETAT = {"cdc": RACINE / "cahiers" / "sae51_equerre.json", "concept": None, "resultat": None}


# ---------- données ----------
def lister_cahiers() -> list[Path]:
    return sorted((RACINE / "cahiers").glob("*.json"))


def lister_concepts() -> list[Path]:
    return sorted((RACINE / "concepts").rglob("*.json"))


def lister_resultats() -> list[Path]:
    return sorted(p for p in (RACINE / "resultats").glob("*") if (p / "etude.db").exists() or (p / "synthese.json").exists())


def relatif(p: Path) -> str:
    return str(p.relative_to(RACINE)) if p.is_relative_to(RACINE) else str(p)


def cdc() -> CahierDesCharges:
    return CahierDesCharges.charger(ETAT["cdc"])


def valeurs_centrales(definition: dict, cahier: CahierDesCharges, graine: int | None = None) -> dict:
    from gabarit.etudes.concept_json import proposer_valeurs
    from gabarit.ia.outils import _Tirage

    return proposer_valeurs(definition, cahier, _Tirage(None if graine is None else np.random.default_rng(graine)))


def verifier_concept(chemin: str, chemin_cdc: str, calcul: bool) -> dict:
    """Exécuté dans un processus séparé (Gmsh n'aime pas les fils d'exécution)."""
    from gabarit.ia.outils import examiner_concept

    definition = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return examiner_concept(definition, CahierDesCharges.charger(chemin_cdc), calcul=calcul)


def exporter_step_fichier(chemin_concept: str, chemin_cdc: str, valeurs: dict, sortie: str) -> str:
    from gabarit.export.step import exporter_step

    c = ConceptJSON.charger(chemin_concept, valeurs, CahierDesCharges.charger(chemin_cdc))
    return str(exporter_step(c, sortie))


def lire_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ecrire_config(**maj) -> None:
    c = {**lire_config(), **maj}
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")


def charger_config() -> None:
    cle = lire_config().get("anthropic_api_key")
    if cle and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = cle


def fmt(x, n=2, unite="") -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{x:,.{n}f}".replace(",", " ").replace(".", ",") + (f" {unite}" if unite else "")


# ---------- page ----------
@ui.page("/")
def page() -> None:
    ui.colors(primary="#2a6fdb")
    ui.query("body").style("background:#f6f7f9")
    with ui.header().classes("items-center bg-white text-black shadow-sm"):
        ui.icon("architecture", size="28px").classes("text-primary")
        ui.label("Gabarit").classes("text-xl font-semibold")
        ui.label("conception de pièces mécano-soudées optimisées").classes("text-gray-500")
        ui.space()
        etiquette_cdc = ui.label().classes("text-sm text-gray-600")
        ui.button(icon="key", on_click=lambda: dialogue_cle.open()).props("flat round").tooltip("Clé API Anthropic")

    def maj_entete():
        etiquette_cdc.text = f"Cahier des charges : {cdc().nom}"
    maj_entete()

    with ui.dialog() as dialogue_cle, ui.card().classes("w-[480px]"):
        ui.label("Clé API Anthropic").classes("text-lg font-semibold")
        ui.label("Sert à l'IA Concepteur. Elle reste sur ce PC.").classes("text-sm text-gray-600")
        champ = ui.input("ANTHROPIC_API_KEY", password=True, password_toggle_button=True,
                         value=os.environ.get("ANTHROPIC_API_KEY", "")).classes("w-full")
        memoriser = ui.checkbox(f"Mémoriser dans {CONFIG}", value=CONFIG.exists())

        def enregistrer_cle():
            os.environ["ANTHROPIC_API_KEY"] = champ.value.strip()
            if memoriser.value:
                ecrire_config(anthropic_api_key=champ.value.strip())
            dialogue_cle.close()
            ui.notify("Clé enregistrée", type="positive")
        with ui.row().classes("w-full justify-end"):
            ui.button("Annuler", on_click=dialogue_cle.close).props("flat")
            ui.button("Enregistrer", on_click=enregistrer_cle)

    with ui.tabs().classes("w-full bg-white") as onglets:
        t_cdc = ui.tab("cdc", "1. Cahier des charges", icon="description")
        t_concepts = ui.tab("concepts", "2. Concepts", icon="view_in_ar")
        t_optim = ui.tab("optim", "3. Optimisation", icon="tune")
        t_res = ui.tab("res", "4. Résultats", icon="insights")
        t_app = ui.tab("app", "5. Apprentissage", icon="school")
    with ui.tab_panels(onglets, value=t_concepts).classes("w-full bg-transparent"):
        with ui.tab_panel(t_cdc):
            onglet_cdc(maj_entete)
        with ui.tab_panel(t_concepts):
            onglet_concepts()
        with ui.tab_panel(t_optim):
            onglet_optimisation()
        with ui.tab_panel(t_res):
            onglet_resultats()
        with ui.tab_panel(t_app):
            onglet_apprentissage()


# ---------- 1. cahier des charges ----------
def onglet_cdc(maj_entete) -> None:
    c = cdc()
    with ui.row().classes("w-full items-center"):
        choix = ui.select({str(p): relatif(p) for p in lister_cahiers()}, value=str(ETAT["cdc"]),
                          label="Fichier").classes("w-96")
        ui.label("Les valeurs ci-dessous servent à l'IA Concepteur, aux contrôles et au calcul.").classes(
            "text-gray-600")

    champs = {}
    with ui.grid(columns=3).classes("w-full gap-4"):
        with ui.card().classes("[&_.q-field]:w-full"):
            ui.label("Matériau").classes("font-semibold")
            champs["materiau.nom"] = ui.input("Nuance", value=c.materiau.nom)
            champs["materiau.Re"] = ui.number("Limite élastique Re (MPa)", value=c.materiau.Re)
            champs["materiau.E"] = ui.number("Module d'Young E (MPa)", value=c.materiau.E)
            champs["materiau.nu"] = ui.number("Coefficient de Poisson", value=c.materiau.nu, step=0.01)
            champs["materiau.rho"] = ui.number("Masse volumique (kg/m³)", value=c.materiau.rho)
        with ui.card().classes("[&_.q-field]:w-full"):
            ui.label("Chargement et critères").classes("font-semibold")
            ch = c.chargements[0]
            champs["force"] = ui.number("Effort (N)", value=ch.force)
            champs["distance"] = ui.number("Distance à l'encastrement (mm)", value=ch.distance_encastrement)
            champs["flambement"] = ui.number("Facteur de flambement minimal",
                                            value=c.criteres.coefficient_flambement_min, step=0.5)
            ui.label(f"Critère : {c.criteres.contrainte}").classes("text-sm text-gray-600")
            ui.label(f"Objectif : {c.criteres.objectif}").classes("text-sm text-gray-600")
        with ui.card().classes("[&_.q-field]:w-full"):
            ui.label("Procédé").classes("font-semibold")
            champs["epaisseurs"] = ui.input("Épaisseurs disponibles (mm)",
                                            value="; ".join(f"{e:g}" for e in c.procede.epaisseurs_disponibles))
            champs["ves"] = ui.input("Vés de pliage (mm)", value="; ".join(f"{v:g}" for v in c.procede.ves_disponibles))
            champs["pieces"] = ui.number("Nombre de pièces maximum", value=c.procede.nombre_pieces_max, step=1)
            ui.label(f"Découpe : {c.procede.decoupe}").classes("text-sm text-gray-600")
            ui.label(f"Pliage : {c.procede.pliage}").classes("text-sm text-gray-600")
        with ui.card().classes("[&_.q-field]:w-full"):
            ui.label("Encombrement").classes("font-semibold")
            champs["y_max"] = ui.number("Demi-largeur entre nervures (mm)", value=c.encombrement.y_max)
            champs["x_max"] = ui.number("Bout de la semelle x max (mm)", value=c.encombrement.x_max)
            champs["z_min"] = ui.number("Dessus de la semelle z min (mm)", value=c.encombrement.z_min)
            ui.label(c.encombrement.description).classes("text-sm text-gray-600")
        with ui.card().classes("col-span-2"):
            ui.label(f"Grille de fixation : {len(c.grille_fixation.trous)} trous, {c.grille_fixation.taraudage}").classes(
                "font-semibold")
            ui.plotly(figure_grille(c)).classes("w-full")
    with ui.card().classes("w-full"):
        ui.label("Hypothèses").classes("font-semibold")
        champs["hypotheses"] = ui.textarea(value="\n".join(c.hypotheses)).classes("w-full").props("autogrow")

    def nombres(texte: str) -> list[float]:
        return [float(x.replace(",", ".")) for x in re.split(r"[;\s]+", texte.strip()) if x]

    def enregistrer():
        d = cdc().model_dump()
        try:
            d["materiau"].update({k.split(".")[1]: champs[k].value for k in champs if k.startswith("materiau.")})
            d["chargements"][0]["force"] = float(champs["force"].value)
            d["chargements"][0]["distance_encastrement"] = float(champs["distance"].value)
            d["criteres"]["coefficient_flambement_min"] = float(champs["flambement"].value)
            d["procede"]["epaisseurs_disponibles"] = nombres(champs["epaisseurs"].value)
            d["procede"]["ves_disponibles"] = nombres(champs["ves"].value)
            d["procede"]["nombre_pieces_max"] = int(champs["pieces"].value)
            d["encombrement"].update(y_max=float(champs["y_max"].value), y_min=-float(champs["y_max"].value),
                                     x_max=float(champs["x_max"].value), z_min=float(champs["z_min"].value))
            d["hypotheses"] = [h for h in champs["hypotheses"].value.splitlines() if h.strip()]
            CahierDesCharges.model_validate(d)
        except (ValueError, TypeError) as e:
            ui.notify(f"Valeur invalide : {e}", type="negative")
            return
        Path(ETAT["cdc"]).write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        maj_entete()
        ui.notify(f"Enregistré dans {relatif(Path(ETAT['cdc']))}", type="positive")

    def changer(e):
        ETAT["cdc"] = Path(e.value)
        ui.navigate.reload()
    choix.on_value_change(changer)
    ui.button("Enregistrer le cahier des charges", icon="save", on_click=enregistrer).classes("mt-2")


# ---------- 2. concepts ----------
def onglet_concepts() -> None:
    concepts = lister_concepts()
    if ETAT["concept"] is None and concepts:
        ETAT["concept"] = concepts[0]
    with ui.row().classes("w-full no-wrap gap-4 items-start"):
        with ui.column().classes("w-[34%] gap-4"):
            with ui.card().classes("w-full"):
                ui.label("Concepts").classes("text-lg font-semibold")
                lignes = []
                for p in concepts:
                    try:
                        d = json.loads(p.read_text(encoding="utf-8"))
                    except ValueError:
                        continue
                    lignes.append({"id": str(p), "nom": d.get("nom", p.stem), "origine": origine(p),
                                   "variables": len(d.get("variables", {})), "toles": len(d.get("toles", []))})
                table = ui.table(rows=lignes, columns=[
                    {"name": "nom", "label": "Concept", "field": "nom", "align": "left"},
                    {"name": "origine", "label": "Origine", "field": "origine"},
                    {"name": "toles", "label": "Tôles", "field": "toles"},
                    {"name": "variables", "label": "Variables", "field": "variables"}],
                    row_key="id", selection="single").classes("w-full").props("flat dense")
                table.selected = [r for r in lignes if r["id"] == str(ETAT["concept"])]
            carte_ia()
        with ui.column().classes("grow gap-4"):
            detail = ui.column().classes("w-full")

    def montrer(chemin: Path | None, graine: int | None = None):
        detail.clear()
        with detail:
            detail_concept(chemin, graine, montrer)

    def selection(e):
        if e.selection:
            ETAT["concept"] = Path(e.selection[0]["id"])
            montrer(ETAT["concept"])
    table.on_select(selection)
    montrer(ETAT["concept"])


def detail_concept(chemin: Path | None, graine: int | None, montrer) -> None:
    if chemin is None:
        ui.label("Aucun concept. Lance l'IA Concepteur ou ajoute un fichier dans concepts/.")
        return
    definition = json.loads(chemin.read_text(encoding="utf-8"))
    cahier = cdc()
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-start"):
            with ui.column().classes("grow gap-0"):
                ui.label(definition.get("nom", chemin.stem)).classes("text-lg font-semibold")
                ui.label(relatif(chemin)).classes("text-xs text-gray-500")
        ui.label(definition.get("description", "")).classes("text-gray-700")
        try:
            valeurs = valeurs_centrales(definition, cahier, graine)
            concept = ConceptJSON(definition, valeurs, cahier)
            ui.plotly(figure_faces(faces_du_concept(concept))).classes("w-full")
            legende = "point central des variables" if graine is None else f"tirage aléatoire n° {graine}"
            ui.label(f"Maquette au {legende}, masse {fmt(concept.masse(cahier.materiau.rho))} kg").classes(
                "text-sm text-gray-600")
        except Exception as e:  # concept mal formé : on l'affiche quand même
            ui.label(f"Géométrie impossible à construire : {e}").classes("text-negative")
        with ui.row():
            ui.button("Point central", icon="center_focus_strong", on_click=lambda: montrer(chemin)).props("outline")
            ui.button("Autre tirage", icon="casino",
                      on_click=lambda: montrer(chemin, int(np.random.default_rng().integers(1, 10_000)))).props("outline")
            bouton_verif = ui.button("Vérifier (géométrie + calcul rapide)", icon="fact_check")
            ui.button("Optimiser ce concept", icon="tune",
                      on_click=lambda: ui.notify("Choisis-le dans l'onglet Optimisation")).props("flat")
        zone_verif = ui.column().classes("w-full")

        async def verifier():
            bouton_verif.disable()
            zone_verif.clear()
            with zone_verif:
                ui.spinner(size="lg")
                ui.label("Contrôle en cours (10 à 40 s)…")
            try:
                r = await run.cpu_bound(verifier_concept, str(chemin), str(ETAT["cdc"]), True)
            finally:
                bouton_verif.enable()
            zone_verif.clear()
            with zone_verif:
                afficher_rapport(r)
        bouton_verif.on_click(verifier)

    with ui.card().classes("w-full"):
        ui.label("Variables (ce que l'optimiseur fait varier)").classes("font-semibold")
        lignes = []
        for nom, s in definition.get("variables", {}).items():
            ty = s.get("type")
            plage = {"epaisseur": "épaisseurs de l'atelier", "fraction": "0 à 1",
                     "choix": ", ".join(map(str, s.get("valeurs", []))),
                     "reel": f"{s.get('min')} à {s.get('max')} mm",
                     "retour": f"absent, ou aile mini à {s.get('max')} mm (tôle {s.get('tole')})"}.get(ty, "")
            lignes.append({"nom": nom, "type": ty, "plage": plage})
        ui.table(rows=lignes, columns=[{"name": k, "label": k.capitalize(), "field": k, "align": "left"}
                                       for k in ("nom", "type", "plage")], row_key="nom").classes("w-full").props(
            "flat dense")
        if definition.get("derivees"):
            with ui.expansion("Cotes dérivées", icon="functions").classes("w-full"):
                for k, v in definition["derivees"].items():
                    ui.label(f"{k} = {v}").classes("font-mono text-sm")
    with ui.card().classes("w-full"):
        ui.label("Justifications des choix de conception").classes("font-semibold")
        for k, v in definition.get("justifications", {}).items():
            with ui.row().classes("no-wrap"):
                ui.label(k).classes("font-semibold min-w-[110px]")
                ui.label(v).classes("text-gray-700")


def afficher_rapport(r: dict) -> None:
    if r.get("erreurs_structure"):
        ui.label("Fichier mal formé :").classes("text-negative font-semibold")
        for e in r["erreurs_structure"]:
            ui.label(f"• {e}")
        return
    centre = r.get("centre", {})
    tir = r.get("tirages_aleatoires", {})
    ok = r.get("valide")
    ui.label("Concept valide" if ok else "Concept à corriger").classes(
        "font-semibold " + ("text-positive" if ok else "text-negative"))
    for e in centre.get("erreurs", []):
        ui.label(f"• {e}")
    ui.label(f"Tirages aléatoires fabricables : {tir.get('valides')} / {tir.get('nombre')}")
    for m, k in tir.get("erreurs_frequentes", [])[:4]:
        ui.label(f"• {m.replace('#', '…')} ({k} fois)").classes("text-sm text-gray-600")
    calc = centre.get("calcul_rapide")
    if calc and "effort_tenu_N" in calc:
        zone = ", ".join(calc["zone_sigma_utile"]["faces"]) or "?"
        ui.label(f"Calcul rapide au point central : masse {fmt(centre.get('masse_kg'))} kg, effort tenu "
                 f"{fmt(calc['effort_tenu_N'] / 1000, 1)} kN (objectif {fmt(calc['objectif_requis_N'] / 1000, 2)} kN), "
                 f"contrainte {calc['sigma_utile_MPa']} MPa vers {zone}, flèche {fmt(calc['fleche_mm'])} mm, "
                 f"vis {fmt(calc['traction_vis_max_N'] / 1000, 1)} kN")
    elif calc:
        ui.label(f"Calcul rapide impossible : {calc.get('erreur')}").classes("text-negative")


def origine(p: Path) -> str:
    return "exploration" if "exploration" in p.parts else "IA" if "ia" in p.parts else "manuel"


def dernier_catalogue() -> list[dict]:
    """Lignes du catalogue de la dernière exploration longue, de la plus récente à la plus ancienne."""
    cats = sorted((RACINE / "resultats" / "exploration").glob("*/catalogue.csv"))
    if not cats:
        return []
    with cats[-1].open(encoding="utf-8-sig") as f:
        lignes = list(reversed(list(csv.DictReader(f))))
    for r in lignes:  # le nom court de la piste (avant les deux-points), ou le concept d'une piste inventée
        inventee = r.get("piste", "").startswith("Invente")
        r["piste_courte"] = ("inventée" + (f" : {r['concept_nom']}" if r.get("concept_nom") else "") if inventee
                             else r.get("piste", "").split(" :")[0])
    return lignes


def carte_ia() -> None:
    tache = TACHES["concepteur"]
    cfg = lire_config()
    with ui.card().classes("w-full"):
        ui.label("IA Concepteur").classes("text-lg font-semibold")
        ui.label("L'IA propose des formes générales, les vérifie et les corrige elle-même.").classes(
            "text-sm text-gray-600")
        moteur = ui.toggle({"claude": "Claude", "ouvert": "Modèle ouvert"}, value=cfg.get("ia_moteur", "claude"))
        with ui.column().classes("w-full gap-0") as zone_ouvert:
            url = ui.input("Adresse du serveur", value=cfg.get("ia_url", "http://localhost:11434/v1")).classes("w-full")
            with ui.row().classes("w-full no-wrap"):
                modele = ui.input("Modèle", value=cfg.get("ia_modele", "qwen3:8b")).classes("w-1/2")
                cle = ui.input("Clé (service hébergé)", password=True,
                               value=os.environ.get("GABARIT_IA_CLE", "")).classes("w-1/2")
            sans_outils = ui.checkbox("Sans appel d'outils (concept rendu dans le texte)",
                                      value=cfg.get("ia_sans_outils", False))
            with ui.row().classes("w-full items-center"):
                explo = ui.checkbox("Exploration longue : une piste différente par essai, pendant",
                                    value=cfg.get("ia_exploration", False))
                heures = ui.number("Heures", value=cfg.get("ia_heures", 8), min=0.1, max=72, step=0.5,
                                   format="%.1f").classes("w-24")
                heures.bind_visibility_from(explo, "value")
            ui.label("Ollama : http://localhost:11434/v1 · LM Studio : http://localhost:1234/v1 · "
                     "OpenRouter : https://openrouter.ai/api/v1 · Mistral : https://api.mistral.ai/v1").classes(
                "text-xs text-gray-500")
        zone_ouvert.bind_visibility_from(moteur, "value", backward=lambda v: v == "ouvert")
        def exploration() -> bool:
            return moteur.value == "ouvert" and explo.value

        with ui.row().classes("w-full") as zone_n:
            n = ui.number("Concepts", value=3, min=1, max=6, step=1).classes("w-24")
            effort = ui.select(["medium", "high", "xhigh"], value="high", label="Effort").classes("w-32")
            effort.bind_visibility_from(moteur, "value", backward=lambda v: v == "claude")
        consigne = ui.textarea("Ton idée ou une consigne (facultatif)").classes("w-full").props("autogrow")
        with ui.row():
            bouton = ui.button("Proposer des concepts", icon="auto_awesome")
            arret = ui.button("Arrêter", icon="stop", on_click=tache.arreter).props("flat color=negative")

        def maj_mode(_=None):
            zone_n.set_visibility(not exploration())
            consigne.set_visibility(not exploration())
            bouton.set_text("Lancer l'exploration" if exploration() else "Proposer des concepts")
        moteur.on_value_change(maj_mode)
        explo.on_value_change(maj_mode)
        maj_mode()
        journal = ui.log(max_lines=400).classes("w-full h-48 text-xs")
        for l in tache.lignes[-200:]:
            journal.push(l)
        with ui.expansion("Catalogue de la dernière exploration", icon="travel_explore").classes("w-full"):
            catalogue = ui.table(rows=dernier_catalogue(), row_key="essai", columns=[
                {"name": c, "label": lab, "field": c, "align": "left" if c == "piste_courte" else "right"}
                for c, lab in (("essai", "N°"), ("piste_courte", "Piste"), ("statut", "Résultat"), ("masse_kg", "kg"),
                               ("effort_tenu_kN", "kN tenus"), ("kg_par_kN", "kg/kN"), ("duree_min", "min"))],
            ).classes("w-full").props("flat dense wrap-cells")
            ui.label("Masse et effort au point central des variables, calcul grossier : un ordre de grandeur pour "
                     "trier les pistes, pas le résultat de l'optimisation.").classes("text-xs text-gray-500")

        def lancer():
            args = ["exemples/concevoir.py", str(int(n.value)), "--cdc", str(ETAT["cdc"])]
            if exploration():
                args = ["exemples/explorer.py", "--heures", f"{float(heures.value or 8):g}", "--cdc", str(ETAT["cdc"])]
            env = {}
            if moteur.value == "claude":
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    ui.notify("Renseigne d'abord la clé API (icône clé en haut à droite)", type="warning")
                    return
                args += ["--effort", effort.value]
            else:
                args += [] if exploration() else ["--moteur", "ouvert"]
                args += ["--url", url.value.strip(), "--modele", modele.value.strip()]
                if sans_outils.value:
                    args.append("--sans-outils")
                if cle.value.strip():
                    os.environ["GABARIT_IA_CLE"] = env["GABARIT_IA_CLE"] = cle.value.strip()
            ecrire_config(ia_moteur=moteur.value, ia_url=url.value.strip(), ia_modele=modele.value.strip(),
                          ia_sans_outils=sans_outils.value, ia_exploration=explo.value, ia_heures=heures.value)
            if consigne.value.strip() and not exploration():
                args += ["--consigne", consigne.value.strip()]
            journal.clear()
            tache.lancer(args, titre="Exploration" if exploration() else "IA Concepteur", env=env)

        def suivre():
            for l in tache.nouvelles_lignes():
                journal.push(l)
                if l.startswith(("[catalogue]", "[fin")):
                    catalogue.rows = dernier_catalogue()
                    catalogue.update()
                if l.startswith("[fin"):
                    ui.notify("IA terminée : appuie sur F5 pour voir les nouveaux concepts", type="info")
            bouton.set_enabled(not tache.en_cours)
            arret.set_visibility(tache.en_cours)
        bouton.on_click(lancer)
        ui.timer(1.0, suivre)


# ---------- 3. optimisation ----------
def onglet_optimisation() -> None:
    tache = TACHES["optimisation"]
    concepts = {str(p): relatif(p) for p in lister_concepts()}
    with ui.row().classes("w-full no-wrap gap-4 items-start"):
        with ui.card().classes("w-[34%]"):
            ui.label("Lancer une optimisation").classes("text-lg font-semibold")
            choix = ui.select(concepts, value=str(ETAT["concept"]) if ETAT["concept"] else None,
                              label="Concept").classes("w-full")
            budget = ui.select(list(BUDGETS), value="Normal (32 + 96 + 40 essais)", label="Budget").classes("w-full")
            paralleles = ui.number("Calculs en parallèle", value=os.cpu_count() or 4, min=1, max=64,
                                   step=1).classes("w-48")
            ui.label("Objectif : masse / effort tenu. Contraintes : pas de plastification sous l'effort, "
                     "flambement ≥ minimum du cahier des charges, traction des vis ≤ 33,4 kN. "
                     "Une étude existante pour ce concept est reprise.").classes("text-sm text-gray-600")
            with ui.row():
                bouton = ui.button("Lancer", icon="play_arrow")
                arret = ui.button("Arrêter", icon="stop", on_click=tache.arreter).props("flat color=negative")
            etat = ui.label().classes("text-sm")
            progression = ui.linear_progress(value=0, show_value=False).classes("w-full")
        with ui.column().classes("grow gap-4"):
            with ui.card().classes("w-full"):
                titre = ui.label("Essais").classes("font-semibold")
                nuage = ui.plotly(figure_essais([])).classes("w-full")
            with ui.card().classes("w-full"):
                journal = ui.log(max_lines=300).classes("w-full h-56 text-xs")
                for l in tache.lignes[-200:]:
                    journal.push(l)

    def total_essais() -> int:
        return sum(int(x) for x in BUDGETS[budget.value].split(","))

    def lancer():
        if not choix.value:
            ui.notify("Choisis un concept", type="warning")
            return
        journal.clear()
        tache.lancer(["exemples/optimiser_concept.py", choix.value, str(int(paralleles.value)),
                      "--budget", BUDGETS[budget.value], "--cdc", str(ETAT["cdc"])], titre=choix.value)
    bouton.on_click(lancer)

    vu = {"n": -1}

    def suivre():
        for l in tache.nouvelles_lignes():
            journal.push(l)
        bouton.set_enabled(not tache.en_cours)
        arret.set_visibility(tache.en_cours)
        cible = Path(tache.titre or choix.value or "")
        if not cible.name:
            return
        essais = lire_essais(RACINE / "resultats" / cible.stem / "essais.csv")
        faisables = [e for e in essais if e["faisable"] == "True"]
        meilleur = min((float(e["objectif_kg_par_kN"]) for e in faisables), default=None)
        if tache.en_cours:
            progression.value = min(1.0, len(essais) / total_essais())
            etat.text = (f"En cours : {len(essais)} essais, {len(faisables)} faisables"
                         + (f", meilleur {fmt(meilleur, 3)} kg/kN" if meilleur else ""))
        elif tache.code is not None:
            progression.value = 1.0
            etat.text = "Terminée : résultats dans l'onglet Résultats" if tache.code == 0 else f"Arrêtée (code {tache.code})"
        titre.text = f"Essais : {relatif(cible) if cible.is_absolute() else cible}"
        if len(essais) != vu["n"]:
            vu["n"] = len(essais)
            nuage.update_figure(figure_essais(essais))
    choix.on_value_change(lambda: vu.update(n=-1))
    ui.timer(2.0, suivre)


# ---------- 4. résultats ----------
def onglet_resultats() -> None:
    dossiers = lister_resultats()
    lignes = []
    for d in dossiers:
        s = lire_synthese(d)
        v4 = (s.get("verification") or {}).get("4 mm", {})
        if not s.get("concept"):  # étude d'une phase antérieure, sans synthèse au format des concepts
            continue
        lignes.append({"id": str(d), "concept": s["concept"], "essais": s.get("essais", "-"),
                       "faisables": s.get("faisables", "-"), "masse": fmt(v4.get("masse_kg")),
                       "tenu": fmt(v4.get("effort_tenu", float("nan")) / 1000, 1),
                       "flambement": fmt(v4.get("facteur_flambement")), "ratio": fmt(v4.get("masse_par_kN"), 3),
                       "_tri": v4.get("masse_par_kN", float("inf"))})
    lignes.sort(key=lambda r: r.pop("_tri"))  # les concepts faisables d'abord, du plus léger par kN tenu
    with ui.card().classes("w-full"):
        ui.label("Comparaison des concepts (vérification sur maillage de 4 mm)").classes("text-lg font-semibold")
        table = ui.table(rows=lignes, row_key="id", selection="single", columns=[
            {"name": "concept", "label": "Concept", "field": "concept", "align": "left"},
            {"name": "essais", "label": "Essais", "field": "essais"},
            {"name": "faisables", "label": "Faisables", "field": "faisables"},
            {"name": "masse", "label": "Masse (kg)", "field": "masse"},
            {"name": "tenu", "label": "Effort tenu (kN)", "field": "tenu"},
            {"name": "flambement", "label": "Flambement", "field": "flambement"},
            {"name": "ratio", "label": "kg / kN", "field": "ratio"}]).classes("w-full").props("flat dense")
    detail = ui.column().classes("w-full")

    def montrer(dossier: Path):
        detail.clear()
        with detail:
            detail_resultat(dossier)

    def selection(e):
        if e.selection:
            montrer(Path(e.selection[0]["id"]))
    table.on_select(selection)
    if lignes:
        table.selected = [lignes[0]]
        montrer(Path(lignes[0]["id"]))


def lire_synthese(dossier: Path) -> dict:
    try:
        return json.loads((dossier / "synthese.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def trouver_concept(s: dict, dossier: Path) -> Path | None:
    if s.get("fichier_concept"):
        p = Path(s["fichier_concept"])
        p = p if p.is_absolute() else RACINE / p
        if p.exists():
            return p
    return next((p for p in lister_concepts() if p.stem == dossier.name), None)


def detail_resultat(dossier: Path) -> None:
    s = lire_synthese(dossier)
    if not s:
        ui.label("Étude sans synthèse (en cours ou interrompue).")
        return
    if "verification" not in s:
        with ui.card().classes("w-full"):
            ui.label(f"{s.get('concept')} : aucun essai faisable sur {s.get('essais')}").classes(
                "text-lg font-semibold text-negative")
            ui.label(f"Meilleur effort tenu : {fmt(s.get('meilleur_effort_tenu_N', 0) / 1000, 1)} kN")
            if (dossier / "essais.png").exists():
                ui.image(dossier / "essais.png").classes("w-[640px]")
        return
    v = s["verification"]
    v4 = v.get("4 mm", next(iter(v.values())))
    with ui.row().classes("w-full gap-4"):
        for titre, valeur in (("Masse", f"{fmt(v4['masse_kg'])} kg"),
                              ("Effort tenu", f"{fmt(v4['effort_tenu'] / 1000, 1)} kN"),
                              ("Flambement", fmt(v4["facteur_flambement"])),
                              ("Masse / effort", f"{fmt(v4['masse_par_kN'], 3)} kg/kN"),
                              ("Traction par vis", f"{fmt(v4['traction_vis_max'] / 1000, 1)} kN")):
            with ui.card().classes("min-w-[150px]"):
                ui.label(titre).classes("text-sm text-gray-600")
                ui.label(valeur).classes("text-2xl font-semibold")
    with ui.row().classes("w-full no-wrap gap-4 items-start"):
        with ui.card().classes("w-[55%]"):
            ui.label(f"{s['concept']} : optimum (essai {s.get('meilleur_essai')})").classes("font-semibold")
            ui.plotly(figure_faces(faces_de_synthese(s["geometrie"]), hauteur=430)).classes("w-full")
            with ui.row():
                bouton_step = ui.button("Télécharger le STEP", icon="download")
                ui.button("Synthèse (JSON)", icon="data_object",
                          on_click=lambda: ui.download(dossier / "synthese.json", f"{dossier.name}_synthese.json")).props(
                    "outline")
        with ui.card().classes("grow"):
            ui.label("Vérification à plusieurs tailles de maille").classes("font-semibold")
            lignes = [{"maille": k, "masse": fmt(x["masse_kg"]), "sigma": fmt(x["sigma_max_utile"], 0),
                       "tenu": fmt(x["effort_tenu"] / 1000, 1), "flamb": fmt(x["facteur_flambement"]),
                       "fleche": fmt(x["fleche"]), "vis": fmt(x["traction_vis_max"] / 1000, 1)} for k, x in v.items()]
            ui.table(rows=lignes, row_key="maille", columns=[
                {"name": "maille", "label": "Maille", "field": "maille"},
                {"name": "sigma", "label": "σ utile (MPa)", "field": "sigma"},
                {"name": "tenu", "label": "Tenu (kN)", "field": "tenu"},
                {"name": "flamb", "label": "Flambement", "field": "flamb"},
                {"name": "fleche", "label": "Flèche (mm)", "field": "fleche"},
                {"name": "vis", "label": "Vis (kN)", "field": "vis"}]).classes("w-full").props("flat dense")
            ui.label("Valeurs des variables").classes("font-semibold mt-2")
            with ui.grid(columns=2).classes("gap-x-6 gap-y-0"):
                for k, x in s["valeurs"].items():
                    ui.label(k).classes("font-mono text-sm")
                    ui.label(fmt(x, 2) if isinstance(x, float) else str(x)).classes("text-sm")
    with ui.row().classes("w-full no-wrap gap-4"):
        for nom in ("optimum_von_mises.png", "essais.png"):
            if (dossier / nom).exists():
                with ui.card().classes("w-1/2"):
                    ui.image(dossier / nom).classes("w-full")

    async def step():
        chemin_concept = trouver_concept(s, dossier)
        if chemin_concept is None:
            ui.notify("Fichier du concept introuvable", type="negative")
            return
        sortie = Path(tempfile.gettempdir()) / f"{dossier.name}_optimum.step"
        bouton_step.disable()
        try:
            await run.cpu_bound(exporter_step_fichier, str(chemin_concept), str(ETAT["cdc"]), s["valeurs"], str(sortie))
        finally:
            bouton_step.enable()
        ui.download(sortie, sortie.name)
    bouton_step.on_click(step)


# ---------- 5. apprentissage ----------
STATUTS_LECONS = {"confirmee": ("confirmée", "positive"), "nuancee": ("nuancée", "warning"),
                  "observee": ("observée", "primary"), "hypothese": ("hypothèse", "grey"),
                  "contredite": ("contredite", "negative")}


def onglet_apprentissage() -> None:
    from gabarit.apprentissage.carnet import CARNET_DEFAUT

    tache = TACHES["apprentissage"]
    cfg = lire_config()
    with ui.row().classes("w-full no-wrap gap-4 items-start"):
        with ui.card().classes("w-[34%]"):
            ui.label("Boucle d'apprentissage").classes("text-lg font-semibold")
            ui.label("Chaque cycle teste une expérience prévue, améliore le meilleur concept (une seule modification, "
                     "prédiction écrite avant le calcul) ou explore un principe nouveau. Gabarit optimise et mesure, "
                     "l'IA Analyste explique les écarts de kg/kN et range des leçons, vérifiées sur les mesures, dans "
                     "le carnet que lira l'IA Concepteur au cycle suivant.").classes("text-sm text-gray-600")
            moteur = ui.toggle({"claude": "Claude", "ouvert": "Modèle ouvert"}, value=cfg.get("ia_moteur", "claude"))
            with ui.column().classes("w-full gap-0") as zone_ouvert:
                url = ui.input("Adresse du serveur", value=cfg.get("ia_url", "http://localhost:11434/v1")).classes("w-full")
                modele = ui.input("Modèle", value=cfg.get("ia_modele", "qwen3:8b")).classes("w-full")
            zone_ouvert.bind_visibility_from(moteur, "value", backward=lambda v: v == "ouvert")
            analyste = ui.select({"": "IA Analyste : la même que le concepteur", "claude": "IA Analyste : Claude",
                                  "aucun": "Sans IA Analyste (verdicts seulement)"},
                                 value=cfg.get("app_analyste", ""), label="Analyse").classes("w-full")
            with ui.row():
                cycles = ui.number("Cycles", value=cfg.get("app_cycles", 3), min=1, max=500, step=1).classes("w-24")
                paralleles = ui.number("Calculs en parallèle", value=os.cpu_count() or 4, min=1, max=64,
                                       step=1).classes("w-40")
            ui.label("Un cycle prend de 5 à 20 minutes (conception, environ 64 essais d'optimisation, fiche à 6 mm "
                     "avec flambement, analyse).").classes("text-xs text-gray-500")
            with ui.row():
                bouton = ui.button("Lancer", icon="school")
                arret = ui.button("Arrêter", icon="stop", on_click=tache.arreter).props("flat color=negative")
            journal = ui.log(max_lines=400).classes("w-full h-64 text-xs")
            for ligne in tache.lignes[-200:]:
                journal.push(ligne)
        zone = ui.column().classes("grow gap-4")

    def afficher_carnet():
        zone.clear()
        with zone:
            carte_carnet(CARNET_DEFAUT)

    def lancer():
        args = ["exemples/apprendre.py", "--cycles", str(int(cycles.value or 1)), "--moteur", moteur.value,
                "--paralleles", str(int(paralleles.value or 4)), "--cdc", str(ETAT["cdc"])]
        if moteur.value == "ouvert":
            args += ["--url", url.value.strip(), "--modele", modele.value.strip()]
        if analyste.value:
            args += ["--analyste", analyste.value]
        if "claude" in (moteur.value, analyste.value or moteur.value) and not os.environ.get("ANTHROPIC_API_KEY"):
            ui.notify("Renseigne d'abord la clé API (icône clé en haut à droite)", type="warning")
            return
        ecrire_config(ia_moteur=moteur.value, ia_url=url.value.strip(), ia_modele=modele.value.strip(),
                      app_analyste=analyste.value, app_cycles=cycles.value)
        journal.clear()
        tache.lancer(args, titre="Apprentissage")
    bouton.on_click(lancer)

    def suivre():
        for ligne in tache.nouvelles_lignes():
            journal.push(ligne)
            if ligne.startswith(("[verdict]", "[fin", "=== Cycle")):
                afficher_carnet()
        bouton.set_enabled(not tache.en_cours)
        arret.set_visibility(tache.en_cours)
    afficher_carnet()
    ui.timer(1.0, suivre)


def carte_carnet(chemin: Path) -> None:
    from gabarit.apprentissage.carnet import Carnet, valeur

    c = Carnet.charger(chemin)
    with ui.card().classes("w-full"):
        ui.label(f"Carnet : {len(c.fiches)} concepts mesurés, {len(c.lecons)} leçons, "
                 f"{sum(e.statut == 'faite' for e in c.experiences)} expériences faites, {c.cycles} cycles").classes(
            "text-lg font-semibold")
        lignes = [{"id": f.id, "concept": f.nom, "faisable": "oui" if f.faisable else "non",
                   "kg": fmt(valeur(f, "kg_par_kN"), 3), "masse": fmt(valeur(f, "masse_kg")),
                   "tenu": fmt(valeur(f, "effort_tenu_N") / 1000, 1), "leq": fmt(valeur(f, "l_eq_mm"), 0),
                   "eta": fmt(valeur(f, "eta"), 3), "flamb": fmt(valeur(f, "flambement")), "limite": f.limite or "-",
                   "parent": f.parent or "-", "_tri": (not f.faisable, valeur(f, "kg_par_kN") if f.mesures else 9e9)}
                  for f in c.fiches]
        lignes.sort(key=lambda r: r.pop("_tri"))
        ui.table(rows=lignes, row_key="id", columns=[
            {"name": k, "label": lab, "field": k, "align": "left" if k in ("concept", "limite") else "right"}
            for k, lab in (("id", "Fiche"), ("concept", "Concept"), ("faisable", "Faisable"), ("kg", "kg/kN"),
                           ("masse", "Masse (kg)"), ("tenu", "Tenu (kN)"), ("leq", "L_eq (mm)"), ("eta", "eta"),
                           ("flamb", "Flambement"), ("limite", "Limite"), ("parent", "Parent"))]).classes(
            "w-full").props("flat dense wrap-cells")
        ui.label("Mesures à 6 mm avec flambement. kg/kN = 1000 rho/Re x L_eq / eta : L_eq mesure le chemin d'effort, "
                 "eta l'utilisation de la matière.").classes("text-xs text-gray-500")
    with ui.card().classes("w-full"):
        ui.label("Leçons").classes("font-semibold")
        for x in sorted(c.lecons, key=lambda x: (list(STATUTS_LECONS).index(x.statut) if x.statut in STATUTS_LECONS
                                                 else 9, x.id)):
            nom, couleur = STATUTS_LECONS.get(x.statut, (x.statut, "grey"))
            with ui.row().classes("w-full no-wrap items-start gap-2"):
                ui.badge(nom, color=couleur).classes("mt-1")
                with ui.column().classes("gap-0"):
                    ui.label(f"{x.id} : {x.enonce}").classes("text-sm")
                    ui.label(x.bilan_preuves() + (f" · {x.mecanisme}" if x.mecanisme else "")).classes(
                        "text-xs text-gray-500")
    if c.experiences:
        with ui.card().classes("w-full"):
            ui.label("Expériences").classes("font-semibold")
            for e in reversed(c.experiences[-20:]):
                ui.label(e.texte()).classes("text-sm")
    stats = c.texte_statistiques()
    if stats:
        with ui.card().classes("w-full"):
            ui.label("Statistiques").classes("font-semibold")
            for t in stats:
                ui.label(t).classes("text-sm")


def lancer() -> None:
    p = argparse.ArgumentParser(description="Gabarit : application locale")
    p.add_argument("--fenetre", action="store_true", help="fenêtre native (nécessite pywebview)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--sans-navigateur", action="store_true", help="ne pas ouvrir le navigateur")
    p.add_argument("--reseau", action="store_true",
                   help="ouvrir le logiciel aux autres appareils du réseau (sinon, ce PC seulement)")
    a = p.parse_args()
    charger_config()
    app.native.window_args["resizable"] = True
    ui.run(title="Gabarit", host="0.0.0.0" if a.reseau else "127.0.0.1", port=a.port, native=a.fenetre, window_size=(1400, 900) if a.fenetre else None,
           show=not a.sans_navigateur, reload=False, language="fr", favicon="📐")
