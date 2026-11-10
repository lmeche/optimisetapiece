"""Carnet d'apprentissage : ce que Gabarit a mesuré, ce qu'il en a tiré, et comment il l'a vérifié.

Trois sortes d'entrées, dans un fichier JSON lisible (apprentissage/carnet.json par défaut) :
- fiches : un concept à ses valeurs optimisées, mesuré par la chaîne complète (compte rendu de
  gabarit/analyse : kg/kN, sa décomposition L_eq / eta, la limite, le point critique, le bilan de chaque face,
  la convergence au maillage) et décrit par des caractéristiques comparables d'un concept à l'autre ;
- leçons : un énoncé, son mécanisme, sa portée, et ses preuves. Les nombres d'une preuve sont toujours lus
  dans les fiches par Gabarit, jamais recopiés d'une IA : une IA désigne des fiches et une grandeur, Gabarit
  va chercher les valeurs et dit si elles vont dans le sens annoncé ;
- expériences : une modification d'un concept parent, avec sa prédiction écrite AVANT le calcul (grandeur,
  sens, seuil). Une fois l'enfant mesuré, Gabarit tranche : confirmée, contredite ou non concluante (écart
  sous le seuil, de l'ordre du bruit du maillage et de l'optimiseur) ;
- prédictions physiques : ce qu'une IA annonce d'un concept avant son calcul (kg/kN, mode de ruine, principes
  de gabarit/savoir cités). Au calcul de la fiche, Gabarit dit si elle est juste, partielle ou fausse, et tient
  pour chaque principe le compte de ses prédictions (calibration : part de justes, biais de l'écart).

Statut d'une leçon, tiré de ses preuves :
- hypothese : proposée, rien ne l'étaye encore ;
- observee : des comparaisons entre concepts vont dans son sens, mais aucune expérience ne l'a testée (deux
  concepts diffèrent toujours par plus d'une chose) ;
- confirmee : au moins une expérience l'a confirmée, aucune ne l'a contredite ;
- contredite : au moins une expérience l'a contredite, aucune ne l'a confirmée ;
- nuancee : des expériences dans les deux sens, la portée est à préciser.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from gabarit.apprentissage import statistiques as st
from gabarit.savoir import prediction as sp

RACINE = Path(__file__).resolve().parents[2]
CARNET_DEFAUT = RACINE / "apprentissage" / "carnet.json"
SEUIL_DEFAUT_PCT = 3.0  # écart sous lequel une expérience n'est pas concluante (bruit du maillage et de l'optimiseur)

# grandeurs mesurées d'une fiche : nom -> (libellé, meilleur sens)
GRANDEURS = {
    "kg_par_kN": ("kg/kN", "baisse"),
    "masse_kg": ("masse (kg)", "baisse"),
    "effort_tenu_N": ("effort tenu (N)", "hausse"),
    "sigma_utile": ("contrainte utile (MPa)", "baisse"),
    "flambement": ("facteur de flambement", "hausse"),
    "traction_vis_N": ("traction max par vis (N)", "baisse"),
    "fleche": ("flèche sous l'appui (mm)", "baisse"),
    "l_eq_mm": ("L_eq, longueur du chemin d'effort (mm)", "baisse"),
    "eta": ("eta, utilisation de la matière", "hausse"),
    "part_flexion": ("part de l'énergie en flexion", "baisse"),
}
METRIQUES_FACE = {
    "part_masse": "part de la masse", "part_energie": "part de l'énergie", "l_eq_mm": "contribution à L_eq (mm)",
    "eta": "utilisation de la matière", "part_flexion": "part de l'énergie en flexion",
    "part_peu_utile": "part de la surface sous 20 % de la contrainte utile",
}
CARACTERISTIQUES = {
    "n_toles": "nombre de tôles", "n_faces": "nombre de faces", "n_plis": "nombre de plis",
    "n_soudures": "nombre de lignes de soudure", "n_doublages": "nombre de doublages",
    "n_retours": "nombre de retours présents", "n_faces_inclinees": "faces ni horizontales ni verticales",
    "n_evidements": "nombre de découpes d'allègement", "aire_evidements_cm2": "aire découpée (cm²)",
    "epaisseur_moy_mm": "épaisseur moyenne (pondérée par l'aire)", "aire_tole_cm2": "aire de tôle (cm²)",
    "part_flexion": "part de l'énergie en flexion", "part_masse_peu_utile": "part de la masse peu utile",
}
STATUTS = ("hypothese", "observee", "confirmee", "contredite", "nuancee")


def _maintenant() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


@dataclass
class Fiche:
    id: str
    nom: str
    fichier: str  # concept JSON (chemin relatif à la racine du dépôt si possible)
    valeurs: dict
    origine: str = "humain"  # humain, ia:<modèle>, eso...
    parent: str | None = None  # fiche dont ce concept dérive (expérience)
    experience: str | None = None  # expérience à laquelle cette fiche répond
    principe: str = ""
    date: str = field(default_factory=_maintenant)
    faisable: bool = False
    erreurs: list[str] = field(default_factory=list)
    taille: float = math.nan  # maille du calcul de la fiche (mm)
    mesures: dict = field(default_factory=dict)  # masse_kg, effort_tenu_N, kg_par_kN, sigma_utile, flambement...
    marges: dict = field(default_factory=dict)  # rapport à chaque limite (plastification, flambement, vis) : > 1 respectée
    decomposition: dict = field(default_factory=dict)  # l_eq_mm, eta, part_flexion
    limite: str = ""
    point_critique: dict = field(default_factory=dict)  # face, texte
    faces: list[dict] = field(default_factory=list)  # bilan compact de chaque face
    caracteristiques: dict = field(default_factory=dict)
    convergence: dict = field(default_factory=dict)  # kg/kN à une autre maille : {"taille": 8, "kg_par_kN": ..}
    optimisation: dict = field(default_factory=dict)  # essais, faisables, budget, maille
    pistes: list[str] = field(default_factory=list)

    def resume(self) -> str:
        if not self.faisable and not self.mesures:
            return f"{self.id} {self.nom} : non calculable ({'; '.join(self.erreurs[:1])})"
        m, d = self.mesures, self.decomposition
        etat = "" if self.faisable else " (NON faisable)"
        effort = m.get("effort_tenu_N")
        flexion = d.get("part_flexion")
        return (f"{self.id} {self.nom}{etat} : {_nombre(m.get('kg_par_kN'), 3)} kg/kN = masse "
                f"{_nombre(m.get('masse_kg'), 2)} kg / {_nombre(effort / 1000 if effort else None, 1)} kN tenus ; "
                f"L_eq {_nombre(d.get('l_eq_mm'), 0)} mm, eta {_nombre(d.get('eta'), 3)}, flexion "
                f"{_nombre(100 * flexion if flexion is not None else None, 0)} % ; flambement "
                f"{_nombre(m.get('flambement'), 2)} ; limite : {self.limite or '?'}")


@dataclass
class Preuve:
    type: str  # comparaison, experience, statistique, observation
    fiches: list[str] = field(default_factory=list)
    grandeur: str = ""
    valeurs: list[float] = field(default_factory=list)  # lues dans les fiches par Gabarit
    attendu: str = ""  # comparaison : « < » ou « > » (fiche 0 par rapport à fiche 1) ; expérience : baisse, hausse
    verdict: str = ""  # conforme, contraire, non_concluante (comparaison) ; confirmee, contredite, non_concluante
    note: str = ""
    date: str = field(default_factory=_maintenant)


@dataclass
class Lecon:
    id: str
    enonce: str
    mecanisme: str = ""
    portee: str = ""
    source: str = "humain"  # humain, analyste:<modèle>, statistique
    statut: str = "hypothese"
    preuves: list[Preuve] = field(default_factory=list)
    creee: str = field(default_factory=_maintenant)
    maj: str = field(default_factory=_maintenant)

    def bilan_preuves(self) -> str:
        exp = [p for p in self.preuves if p.type == "experience"]
        comp = [p for p in self.preuves if p.type == "comparaison"]
        morceaux = []
        if exp:
            morceaux.append(f"{sum(p.verdict == 'confirmee' for p in exp)} expérience(s) confirmée(s), "
                            f"{sum(p.verdict == 'contredite' for p in exp)} contredite(s)")
        if comp:
            morceaux.append(f"{sum(p.verdict == 'conforme' for p in comp)}/{len(comp)} comparaison(s) dans son sens")
        obs = [p for p in self.preuves if p.type == "observation"]
        if obs:
            avis = [f"{n} {quoi}" for n, quoi in ((sum(p.verdict == "conforme" for p in obs), "dans son sens"),
                                                 (sum(p.verdict == "contraire" for p in obs), "contre"),
                                                 (sum(not p.verdict for p in obs), "non jugée(s)")) if n]
            morceaux.append(f"{len(obs)} observation(s) écrite(s) à la main ({', '.join(avis)})")
        return ", ".join(morceaux) or "aucune preuve"


def texte_preuve(p: Preuve) -> str:
    """Une preuve en une ligne : valeurs lues par Gabarit pour une comparaison ou une expérience, note sinon."""
    if p.type == "observation" or not p.valeurs:
        t = p.type + (f" ({', '.join(p.fiches)})" if p.fiches else "") + f", {p.verdict or 'non jugée'}"
    else:
        t = (f"{p.type} {', '.join(p.fiches)} : {p.grandeur} " + " / ".join(_nombre(v, 4) for v in p.valeurs)
             + f" ({p.verdict or p.attendu})")
    return t + (f" : {p.note}" if p.note else "")


@dataclass
class Experience:
    id: str
    parent: str  # fiche de départ
    modification: str
    grandeur: str = "kg_par_kN"
    sens: str = "baisse"  # baisse ou hausse
    seuil_pct: float = SEUIL_DEFAUT_PCT
    lecon: str | None = None
    source: str = ""
    statut: str = "a_faire"  # a_faire, faite, abandonnee
    enfant: str | None = None
    verdict: str = ""  # confirmee, contredite, non_concluante, echec
    mesure: dict = field(default_factory=dict)  # parent, enfant, ecart_pct
    creee: str = field(default_factory=_maintenant)

    def texte(self) -> str:
        pred = f"{GRANDEURS.get(self.grandeur, (self.grandeur,))[0]} en {self.sens} d'au moins {self.seuil_pct:g} %"
        t = (f"{self.id} (depuis {self.parent}{', leçon ' + self.lecon if self.lecon else ''}) : "
             f"{self.modification.rstrip(' .;')} ; prédiction : {pred}")
        if self.statut == "faite":
            m = self.mesure
            ecart = m.get("ecart_pct")
            t += (f" -> {self.enfant} : {_nombre(m.get('parent'), 4)} -> {_nombre(m.get('enfant'), 4)} "
                  f"({'-' if ecart is None else f'{ecart:+.1f} %'}), {self.verdict.replace('_', ' ')}"
                  + (f" ({m['note']})" if m.get("note") else ""))
        elif self.statut == "abandonnee":
            t += f" -> abandonnée ({self.mesure.get('note') or self.verdict or 'pas de concept mesurable'})"
        return t


@dataclass
class Prediction:
    id: str
    concept: str  # fichier du concept (chemin relatif à la racine du dépôt si possible)
    nom: str
    kg_par_kN: float
    mode_limite: str
    principes: list[str] = field(default_factory=list)
    face_critique: str = ""  # face prédite pour le point critique (facultatif)
    h_mm: float | None = None
    effort_tenu_kN: float | None = None
    raisonnement: str = ""
    source: str = ""
    experience: str | None = None
    statut: str = "a_verifier"  # a_verifier, verifiee, abandonnee
    fiche: str | None = None
    verdict: str = ""  # juste, partielle, fausse
    mesure: dict = field(default_factory=dict)  # kg_par_kN, mode_limite, ecart_pct, valeur_juste, mode_juste,
    # face_critique et lieu_juste si une face était prédite, note
    erreurs: list[str] = field(default_factory=list)  # ce qui était illisible dans la prédiction de l'IA
    creee: str = field(default_factory=_maintenant)

    def texte(self) -> str:
        t = (f"{self.id} ({self.nom}{', ' + ', '.join(self.principes) if self.principes else ''}) : prédit "
             f"{_nombre(self.kg_par_kN, 3)} kg/kN, limite {self.mode_limite or '?'}"
             + (f", point critique sur {self.face_critique}" if self.face_critique else ""))
        if self.statut == "verifiee":
            m = self.mesure
            ecart = m.get("ecart_pct")
            t += (f" -> {self.fiche} : {_nombre(m.get('kg_par_kN'), 3)} kg/kN, limite {m.get('mode_limite') or '?'}"
                  + (f", point critique sur {m.get('face_critique') or '?'} (lieu {'juste' if m.get('lieu_juste') else 'faux'})"
                     if "lieu_juste" in m else "")
                  + f" (écart {'-' if ecart is None else f'{ecart:+.0f} %'}), {self.verdict}")
        elif self.statut == "abandonnee":
            t += f" -> non vérifiée ({self.mesure.get('note') or 'pas de fiche mesurable'})"
        return t


def valeur(fiche: Fiche, grandeur: str) -> float:
    """Valeur d'une grandeur dans une fiche : mesure, décomposition ou « face:<id>:<métrique> »."""
    if grandeur.startswith("face:"):
        _, fid, metrique = grandeur.split(":", 2)
        f = next((f for f in fiche.faces if f["id"] == fid), None)
        return float(f.get(metrique, math.nan)) if f else math.nan
    if grandeur in fiche.decomposition:
        return float(fiche.decomposition[grandeur])
    v = fiche.mesures.get(grandeur, math.nan)
    return float(v) if isinstance(v, (int, float)) and v is not None else math.nan


def marge_min(fiche: Fiche) -> tuple[float, str]:
    """Plus petite marge d'une fiche et la limite correspondante ; nan si une marge manque (flambement non
    calculé, fiche sans mesures)."""
    if not fiche.marges:
        return math.nan, "?"
    manquantes = [k for k, v in fiche.marges.items() if v is None or not math.isfinite(float(v))]
    if manquantes:
        return math.nan, ", ".join(manquantes)
    k = min(fiche.marges, key=lambda c: fiche.marges[c])
    return float(fiche.marges[k]), k


def grandeur_valide(grandeur: str) -> bool:
    if grandeur.startswith("face:"):
        morceaux = grandeur.split(":", 2)
        return len(morceaux) == 3 and morceaux[2] in METRIQUES_FACE
    return grandeur in GRANDEURS


def verdict(parent: float, enfant: float, sens: str, seuil_pct: float) -> tuple[str, float]:
    """Verdict d'une prédiction « sens » (baisse ou hausse) d'au moins seuil_pct % entre parent et enfant."""
    if not (math.isfinite(parent) and math.isfinite(enfant)) or parent == 0:
        return "non_concluante", math.nan
    ecart = 100 * (enfant - parent) / abs(parent)
    signe = -1 if sens == "baisse" else 1
    if signe * ecart >= seuil_pct:
        return "confirmee", ecart
    if signe * ecart <= -seuil_pct:
        return "contredite", ecart
    return "non_concluante", ecart


class Carnet:
    def __init__(self, chemin: Path = CARNET_DEFAUT):
        self.chemin = Path(chemin)
        self.fiches: list[Fiche] = []
        self.lecons: list[Lecon] = []
        self.experiences: list[Experience] = []
        self.predictions: list[Prediction] = []
        self.journal: list[dict] = []
        self.cycles = 0  # cycles de la boucle d'apprentissage déjà faits (rythme de l'exploration)

    # ---------- fichier ----------
    @classmethod
    def charger(cls, chemin: Path = CARNET_DEFAUT) -> Carnet:
        c = cls(chemin)
        if not c.chemin.exists():
            return c
        d = json.loads(c.chemin.read_text(encoding="utf-8"))
        c.fiches = [Fiche(**_garder(Fiche, f)) for f in d.get("fiches", [])]
        c.lecons = []
        for x in d.get("lecons", []):
            x = _garder(Lecon, x)
            x["preuves"] = [Preuve(**_garder(Preuve, p)) for p in x.get("preuves", [])]
            c.lecons.append(Lecon(**x))
        c.experiences = [Experience(**_garder(Experience, e)) for e in d.get("experiences", [])]
        c.predictions = [Prediction(**_garder(Prediction, p)) for p in d.get("predictions", [])]
        c.journal = list(d.get("journal", []))
        c.cycles = int(d.get("cycles", 0))
        return c

    def enregistrer(self) -> Path:
        """Écrit le carnet (JSON) et sa version lisible (même nom, .md)."""
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        d = {"version": 1, "cycles": self.cycles, "fiches": [asdict(f) for f in self.fiches],
             "lecons": [asdict(x) for x in self.lecons], "experiences": [asdict(e) for e in self.experiences],
             "predictions": [asdict(p) for p in self.predictions], "journal": self.journal[-500:]}
        for chemin, texte in ((self.chemin, json.dumps(d, indent=1, ensure_ascii=False, default=_json) + "\n"),
                              (self.chemin.with_suffix(".md"), self.texte())):
            tmp = chemin.with_name(chemin.name + ".tmp")
            tmp.write_text(texte, encoding="utf-8")
            tmp.replace(chemin)  # jamais de carnet à moitié écrit
        return self.chemin

    def noter(self, evenement: str) -> None:
        self.journal.append({"date": _maintenant(), "evenement": evenement})

    # ---------- accès ----------
    def fiche(self, ident: str) -> Fiche:
        return next(f for f in self.fiches if f.id == ident)

    def lecon(self, ident: str) -> Lecon:
        return next(x for x in self.lecons if x.id == ident)

    def experience(self, ident: str) -> Experience:
        return next(e for e in self.experiences if e.id == ident)

    def prediction(self, ident: str) -> Prediction:
        return next(p for p in self.predictions if p.id == ident)

    def a_fiche(self, ident: str) -> bool:
        return any(f.id == ident for f in self.fiches)

    def meilleures(self, n: int = 3) -> list[Fiche]:
        ok = [f for f in self.fiches if f.faisable and math.isfinite(valeur(f, "kg_par_kN"))]
        return sorted(ok, key=lambda f: valeur(f, "kg_par_kN"))[:n]

    def experiences_a_faire(self) -> list[Experience]:
        return [e for e in self.experiences if e.statut == "a_faire"]

    def _id(self, prefixe: str, existants: list) -> str:
        n = max((int(x.id[len(prefixe):]) for x in existants if x.id[len(prefixe):].isdigit()), default=0)
        return f"{prefixe}{n + 1:03d}"

    # ---------- ajouts ----------
    def ajouter_fiche(self, cr, concept=None, fichier: str | Path = "", origine: str = "humain",
                      parent: str | None = None, experience: str | None = None, optimisation: dict | None = None,
                      convergence: dict | None = None) -> Fiche:
        """Fiche tirée d'un compte rendu (gabarit.analyse.compte_rendu.CompteRendu) et du concept calculé."""
        f = Fiche(self._id("F", self.fiches), cr.concept, _relatif(fichier), dict(cr.valeurs), origine=origine,
                  parent=parent, experience=experience, taille=cr.taille, erreurs=list(cr.erreurs),
                  optimisation=dict(optimisation or {}), convergence=dict(convergence or {}))
        if concept is not None and hasattr(concept, "definition"):
            f.principe = str(concept.definition.get("description", ""))[:400]
            f.caracteristiques = caracteristiques(concept)
        if cr.valide:
            m = cr.mesures
            f.mesures = {k: _propre(m.get(k)) for k in ("masse_kg", "effort_tenu_N", "kg_par_kN", "sigma_utile",
                                                         "flambement", "traction_vis_N", "fleche")}
            f.decomposition = {"l_eq_mm": _propre(cr.l_eq_mm), "eta": _propre(cr.eta),
                               "part_flexion": _propre(cr.part_flexion)}
            f.marges = {k: _propre(v) for k, v in cr.marges.items()}
            f.limite = cr.limite
            f.point_critique = {"face": cr.point_critique.get("face", ""), "texte": cr.point_critique.get("texte", "")}
            f.faces = [{"id": b.id, "t": b.t, "part_masse": _propre(b.part_masse),
                        "part_energie": _propre(b.part_energie), "l_eq_mm": _propre(b.l_eq_mm), "eta": _propre(b.eta),
                        "part_flexion": _propre(b.part_flexion), "part_peu_utile": _propre(b.part_peu_utile)}
                       for b in cr.faces]
            f.pistes = list(cr.pistes)
            f.caracteristiques.update({
                "part_flexion": _propre(cr.part_flexion),
                "part_masse_peu_utile": _propre(sum(b.part_masse * b.part_peu_utile for b in cr.faces)),
            })
            marges = [v for v in cr.marges.values() if isinstance(v, (int, float)) and math.isfinite(v)]
            f.faisable = bool(marges) and min(marges) >= 1.0 - 1e-9 and math.isfinite(m.get("flambement", math.nan))
        self.fiches.append(f)
        self.noter(f"fiche {f.id} : {f.resume()}")
        return f

    def ajouter_lecon(self, enonce: str, mecanisme: str = "", portee: str = "", source: str = "humain",
                      statut: str = "hypothese") -> Lecon:
        x = Lecon(self._id("L", self.lecons), enonce.strip(), mecanisme.strip(), portee.strip(), source,
                  statut if statut in STATUTS else "hypothese")
        self.lecons.append(x)
        self.noter(f"leçon {x.id} ({x.statut}) : {x.enonce}")
        return x

    def ajouter_experience(self, parent: str, modification: str, grandeur: str = "kg_par_kN", sens: str = "baisse",
                           lecon: str | None = None, seuil_pct: float = SEUIL_DEFAUT_PCT, source: str = "") -> Experience:
        if not self.a_fiche(parent):
            raise ValueError(f"fiche parent inconnue : {parent}")
        if not grandeur_valide(grandeur):
            raise ValueError(f"grandeur inconnue : {grandeur}")
        if sens not in ("baisse", "hausse"):
            raise ValueError(f"sens attendu : baisse ou hausse, pas {sens}")
        if lecon is not None and not any(x.id == lecon for x in self.lecons):
            lecon = None
        e = Experience(self._id("E", self.experiences), parent, modification.strip(), grandeur, sens,
                       float(seuil_pct), lecon, source)
        self.experiences.append(e)
        self.noter(f"expérience {e.id} prévue : {e.texte()}")
        return e

    def ajouter_prediction(self, p: dict, concept: str | Path = "", nom: str = "", source: str = "",
                           experience: str | None = None, erreurs: list[str] | None = None) -> Prediction:
        """Prédiction physique lue par gabarit.savoir.prediction.lire, enregistrée AVANT le calcul du concept."""
        if not (isinstance(p.get("kg_par_kN"), (int, float)) and p["kg_par_kN"] > 0):
            raise ValueError("prédiction sans kg/kN")
        x = Prediction(self._id("PR", self.predictions), _relatif(concept), nom or Path(str(concept)).stem,
                       float(p["kg_par_kN"]), str(p.get("mode_limite") or ""), list(p.get("principes") or []),
                       str(p.get("face_critique") or ""), p.get("h_mm"), p.get("effort_tenu_kN"),
                       str(p.get("raisonnement") or ""), source, experience, erreurs=list(erreurs or []))
        self.predictions.append(x)
        self.noter(f"prédiction {x.id} enregistrée avant le calcul : {x.texte()}")
        return x

    def conclure_prediction(self, ident: str, fiche: str | None, note: str = "") -> Prediction:
        """Verdict d'une prédiction face à la fiche du concept (gabarit.savoir.prediction.juger). Une fiche sans
        mesures (concept non construit ou non calculé) laisse la prédiction non vérifiée : ce n'est pas un résultat
        physique."""
        x = self.prediction(ident)
        x.fiche = fiche
        f = self.fiche(fiche) if fiche and self.a_fiche(fiche) else None
        if f is None or not f.mesures:
            x.statut, x.verdict = "abandonnee", ""
            x.mesure = {"note": note or ("; ".join(f.erreurs[:2]) if f else "aucune fiche mesurable")}
            self.noter(f"prédiction {x.id} non vérifiée : {x.mesure['note']}")
            return x
        x.statut = "verifiee"
        x.verdict, mesure = sp.juger(x.kg_par_kN, x.mode_limite, valeur(f, "kg_par_kN"), sp.mode_de_limite(f.limite),
                                     face_predite=x.face_critique, face_mesuree=str(f.point_critique.get("face", "")))
        x.mesure = {k: (_propre(v) if isinstance(v, float) else v) for k, v in mesure.items()}
        if note:
            x.mesure["note"] = note
        self.noter(f"prédiction {x.id} : {x.texte()}")
        return x

    def calibration(self) -> dict:
        """Pour chaque principe cité (et pour l'ensemble, clé « toutes ») : nombre de prédictions vérifiées, de
        justes, de modes justes, écart moyen signé (biais : > 0 si l'IA prévoit trop lourd) et moyen absolu (%)."""
        faites = [p for p in self.predictions if p.statut == "verifiee"]
        groupes = {"toutes": faites}
        for p in faites:
            for i in p.principes:
                groupes.setdefault(i, []).append(p)
        sortie = {}
        for cle, ps in groupes.items():
            ecarts = [p.mesure.get("ecart_pct") for p in ps]
            ecarts = [e for e in ecarts if isinstance(e, (int, float)) and math.isfinite(e)]
            sortie[cle] = {"n": len(ps), "justes": sum(p.verdict == "juste" for p in ps),
                           "partielles": sum(p.verdict == "partielle" for p in ps),
                           "modes_justes": sum(bool(p.mesure.get("mode_juste")) for p in ps),
                           "lieux_predits": sum("lieu_juste" in p.mesure for p in ps),
                           "lieux_justes": sum(bool(p.mesure.get("lieu_juste")) for p in ps),
                           "ecart_moyen_pct": float(np.mean(ecarts)) if ecarts else math.nan,
                           "ecart_abs_moyen_pct": float(np.mean(np.abs(ecarts))) if ecarts else math.nan}
        return sortie

    def texte_calibration(self) -> list[str]:
        cal = self.calibration()
        if not cal["toutes"]["n"]:
            return []

        def ligne(nom: str, c: dict) -> str:
            return (f"{nom} : {c['n']} prédiction(s) vérifiée(s), {c['justes']} juste(s), {c['partielles']} "
                    f"partielle(s), mode juste {c['modes_justes']}/{c['n']}"
                    + (f", lieu juste {c['lieux_justes']}/{c['lieux_predits']}" if c["lieux_predits"] else "")
                    + f", écart moyen {_signe(c['ecart_moyen_pct'])} (absolu {_nombre(c['ecart_abs_moyen_pct'], 0)} %)")

        from gabarit.savoir.principes import principe

        lignes = [ligne("Toutes", cal["toutes"])]
        for cle in sorted((k for k in cal if k != "toutes"), key=lambda k: int(k[1:])):
            lignes.append(ligne(f"{cle} ({principe(cle).titre})", cal[cle]))
        return lignes

    def preuve_comparaison(self, lecon: Lecon, a: str, b: str, grandeur: str, attendu: str, note: str = "") -> Preuve:
        """Comparaison entre deux fiches : Gabarit lit les deux valeurs et dit si elles vont dans le sens annoncé
        (« < » : la fiche a doit avoir la plus petite valeur). Sous SEUIL_DEFAUT_PCT d'écart : non concluante."""
        va, vb = valeur(self.fiche(a), grandeur), valeur(self.fiche(b), grandeur)
        sens = "baisse" if attendu == "<" else "hausse"
        v, _ = verdict(vb, va, sens, SEUIL_DEFAUT_PCT)  # de b vers a
        p = Preuve("comparaison", [a, b], grandeur, [_propre(va), _propre(vb)], attendu,
                   {"confirmee": "conforme", "contredite": "contraire"}.get(v, "non_concluante"), note)
        lecon.preuves.append(p)
        self.mettre_a_jour_statut(lecon)
        return p

    def conclure_experience(self, ident: str, enfant: str | None, note: str = "") -> Experience:
        """Range le verdict d'une expérience une fois l'enfant mesuré.

        - Pas d'enfant, ou enfant non calculable : expérience abandonnée (échec de construction ou de calcul,
          pas un résultat physique).
        - Sinon la grandeur prédite est lue dans les deux fiches et comparée au seuil de l'expérience.
        - Pour le kg/kN, une pièce nettement non faisable (une marge sous 1 - seuil) compte comme infiniment
          mauvaise : elle ne peut pas être « meilleure » ; à moins du seuil de sa limite, l'écart est de l'ordre
          du bruit du maillage et le verdict porte sur le kg/kN mesuré, avec une note. Une marge inconnue
          (flambement non calculé) rend l'expérience non concluante.
        - Pour le kg/kN, un écart ne compte que s'il dépasse le seuil dans le même sens à la maille des fiches et à
          celle de l'optimiseur (kg/kN de convergence des deux fiches, quand ils existent) : sinon il dépend de
          la maille et l'expérience est non concluante, avec une note."""
        e = self.experience(ident)
        e.enfant = enfant
        fe = self.fiche(enfant) if enfant and self.a_fiche(enfant) else None
        if fe is None or not fe.mesures:
            e.statut, e.verdict = "abandonnee", "echec"
            e.mesure = {"note": note or ("; ".join(fe.erreurs[:2]) if fe else "aucun concept mesurable")}
            self.noter(f"expérience {e.id} abandonnée : {e.mesure['note']}")
            return e
        e.statut = "faite"
        fp = self.fiche(e.parent)
        vp, ve = valeur(fp, e.grandeur), valeur(fe, e.grandeur)
        e.verdict, ecart = verdict(vp, ve, e.sens, e.seuil_pct)
        notes, hors, inconnue = [], {}, False
        for nom, f in (("parent", fp), ("enfant", fe)):
            m, lim = marge_min(f)
            hors[nom] = not f.faisable and math.isfinite(m) and m < 1 - e.seuil_pct / 100
            if not f.faisable:
                inconnue = inconnue or not math.isfinite(m)
                notes.append(f"{nom} non faisable (" + (f"{lim} x{m:.3f}" if math.isfinite(m) else f"marge {lim} inconnue") + ")")
        par_faisabilite = False
        if e.grandeur == "kg_par_kN" and notes:
            if inconnue:
                e.verdict = "non_concluante"
                notes.append("marge inconnue (flambement non calculé ?)")
            elif hors["enfant"] != hors["parent"]:  # une seule des deux pièces est admissible : elle est la meilleure
                par_faisabilite = True
                mieux = hors["parent"]  # l'enfant est la pièce admissible
                e.verdict = "confirmee" if mieux == (e.sens == "baisse") else "contredite"
                notes.append("verdict tiré de la faisabilité : une pièce qui ne tient pas n'est pas meilleure, "
                             "quel que soit son kg/kN")
            else:
                notes.append("à moins du seuil de sa limite : verdict sur le kg/kN mesuré" if not any(hors.values())
                             else "parent et enfant non faisables : verdict sur le kg/kN mesuré")
        e.mesure = {"parent": _propre(vp), "enfant": _propre(ve), "ecart_pct": _propre(ecart)}
        if e.grandeur == "kg_par_kN" and e.verdict in ("confirmee", "contredite") and not par_faisabilite:
            # Un écart ne compte que s'il tient aux deux mailles (fiche et optimiseur) : le bruit du maillage
            # d'un concept à l'autre est de quelques % (voir statistiques).
            cp, ce = fp.convergence or {}, fe.convergence or {}
            kp, ke = (_nombre_ou_nan(c.get("kg_par_kN")) for c in (cp, ce))
            if cp.get("taille") and cp.get("taille") == ce.get("taille") and math.isfinite(kp) and math.isfinite(ke):
                v2, ecart2 = verdict(kp, ke, e.sens, e.seuil_pct)
                e.mesure.update(taille_controle=cp["taille"], ecart_controle_pct=_propre(ecart2))
                if v2 != e.verdict:
                    notes.append(f"à {cp['taille']:g} mm l'écart est de {ecart2:+.1f} % ({v2.replace('_', ' ')}) : "
                                 "il dépend de la maille, verdict non concluant")
                    e.verdict = "non_concluante"
        if notes:
            e.mesure["note"] = " ; ".join(notes)
        if e.lecon:
            x = self.lecon(e.lecon)
            x.preuves.append(Preuve("experience", [e.parent, enfant], e.grandeur,
                                    [e.mesure["parent"], e.mesure["enfant"]], e.sens, e.verdict,
                                    f"{e.id} : {e.modification}"))
            self.mettre_a_jour_statut(x)
        self.noter(f"expérience {e.id} : {e.texte()}")
        return e

    def mettre_a_jour_statut(self, x: Lecon) -> None:
        """Statut tiré des preuves (voir l'en-tête du module). Seules les expériences confirment ou contredisent :
        deux concepts comparés diffèrent par plusieurs choses, une comparaison ne fait qu'« observer »."""
        exp = [p.verdict for p in x.preuves if p.type == "experience"]
        oui, non = exp.count("confirmee"), exp.count("contredite")
        pour = sum(p.verdict == "conforme" for p in x.preuves if p.type in ("comparaison", "observation"))
        contre = sum(p.verdict == "contraire" for p in x.preuves if p.type in ("comparaison", "observation"))
        if oui and non:
            x.statut = "nuancee"
        elif oui:
            x.statut = "confirmee"
        elif non:
            x.statut = "contredite"
        else:
            x.statut = "observee" if pour > contre else "hypothese"
        x.maj = _maintenant()

    # ---------- statistiques ----------
    def statistiques(self) -> dict:
        """Ce que disent les fiches faisables, toutes ensemble : part de L_eq et de eta dans les écarts de kg/kN,
        caractéristiques liées au kg/kN (Spearman, valeur p par permutation, corrigée de Holm), et bruit du
        maillage (décalage du kg/kN entre la maille de l'optimiseur et celle des fiches, d'un concept à l'autre)."""
        ok = [f for f in self.fiches if f.faisable and all(math.isfinite(valeur(f, g)) and valeur(f, g) > 0
                                                            for g in ("kg_par_kN", "l_eq_mm", "eta"))]
        sortie: dict = {"n": len(ok)}
        maillage = self._bruit_maillage()
        if maillage:
            sortie["maillage"] = maillage
        if len(ok) < 2:
            return sortie
        k = [valeur(f, "kg_par_kN") for f in ok]
        sortie["variance"] = st.decomposition_variance(k, [valeur(f, "l_eq_mm") for f in ok], [valeur(f, "eta") for f in ok])
        if len(ok) < 5:
            return sortie
        liens = []
        for c in CARACTERISTIQUES:
            x = [f.caracteristiques.get(c, math.nan) for f in ok]
            if any(v is None or not math.isfinite(float(v)) for v in x) or len(set(x)) < 2:
                continue
            rho, p = st.spearman_p([float(v) for v in x], k)
            liens.append({"caracteristique": c, "rho": rho, "p": p})
        for lien, p in zip(liens, st.holm([lien["p"] for lien in liens])):
            lien["p_holm"] = p
        sortie["liens"] = sorted(liens, key=lambda lien: lien["p"])
        return sortie

    def _bruit_maillage(self) -> dict:
        """Écart-type, d'un concept à l'autre, de ln(kg/kN fiche / kg/kN optimiseur), sur le couple de mailles le
        plus fréquent. Deux concepts dont les kg/kN diffèrent de moins d'environ 2 x racine(2) écarts-types
        peuvent s'inverser avec la maille : c'est l'ordre de grandeur sous lequel une différence n'est pas sûre."""
        groupes: dict[tuple, list[float]] = {}
        for f in self.fiches:
            k, kc, tc = valeur(f, "kg_par_kN"), f.convergence.get("kg_par_kN"), f.convergence.get("taille")
            if math.isfinite(k) and k > 0 and kc and kc > 0 and tc and tc != f.taille:
                groupes.setdefault((f.taille, tc), []).append(math.log(k / kc))
        if not groupes:
            return {}
        (taille, taille_conv), d = max(groupes.items(), key=lambda g: len(g[1]))
        if len(d) < 3:
            return {}
        s = float(np.std(d, ddof=1))
        return {"n": len(d), "taille": taille, "taille_conv": taille_conv,
                "decalage_moyen_pct": 100 * (math.exp(float(np.mean(d))) - 1), "ecart_type_pct": 100 * s,
                "ecart_sur_pct": 100 * 2 * math.sqrt(2) * s}

    def texte_statistiques(self) -> list[str]:
        """Les statistiques en phrases (pour le prompt des IA et le carnet lisible)."""
        s = self.statistiques()
        lignes = []
        v = s.get("variance")
        if v and s["n"] >= 3 and math.isfinite(v.get("l_eq", math.nan)):
            lignes.append(f"Sur {s['n']} concepts faisables, la variance de ln(kg/kN) vient pour {100 * v['l_eq']:.0f} % du "
                          f"chemin d'effort (L_eq), pour {100 * v['eta']:.0f} % de l'utilisation de la matière (eta) et pour "
                          f"{100 * v['covariance']:+.0f} % de leur covariance (identité vérifiée à {v.get('ecart_identite', 0):.0e}).")
        if "liens" in s:
            forts = [lien for lien in s["liens"] if math.isfinite(lien["p_holm"]) and lien["p_holm"] < 0.1]
            lignes += [f"{CARACTERISTIQUES[lien['caracteristique']]} : Spearman {lien['rho']:+.2f} avec le kg/kN, "
                       f"p {lien['p']:.3f}, p corrigé (Holm, {len(s['liens'])} caractéristiques testées) "
                       f"{lien['p_holm']:.3f}" for lien in forts]
            if not forts:
                lignes.append(f"Aucune des {len(s['liens'])} caractéristiques testées n'est liée au kg/kN de façon sûre "
                              "(p corrigé de Holm sous 0,1) : les écarts viennent d'autre chose que ces nombres.")
        m = s.get("maillage")
        if m:
            lignes.append(f"Bruit du maillage : le kg/kN à {m['taille']:g} mm s'écarte de celui à {m['taille_conv']:g} mm "
                          f"de {m['decalage_moyen_pct']:+.1f} % en moyenne, avec {m['ecart_type_pct']:.1f} % d'écart-type "
                          f"d'un concept à l'autre ({m['n']} concepts) : deux concepts à moins d'environ "
                          f"{m['ecart_sur_pct']:.0f} % l'un de l'autre peuvent s'inverser avec la maille.")
        return lignes

    # ---------- textes ----------
    def pour_prompt(self, compact: bool = False, parent: str | None = None) -> str:
        """Section « Ce qu'on a appris » du prompt de l'IA Concepteur, tirée du carnet. compact : pour les petits
        modèles à contexte court (moins de leçons, pas de statistiques)."""
        if not self.fiches and not self.lecons:
            return ""
        n_lecons = 8 if compact else 30
        lignes = [(f"## Ce qu'on a appris sur ce cas (carnet de Gabarit : {len(self.fiches)} concept(s) mesuré(s), "
                   f"{len(self.lecons)} leçon(s), {sum(e.statut == 'faite' for e in self.experiences)} expérience(s) "
                   "faite(s))"),
                  ("Les nombres viennent des calculs de Gabarit, pas d'une IA. Une leçon « confirmée » a passé au moins "
                   "une expérience (un concept modifié sur un seul point, prédiction écrite avant le calcul).")]
        ordre = {"confirmee": 0, "nuancee": 1, "observee": 2, "hypothese": 3, "contredite": 4}
        titres = {"confirmee": "Confirmées par l'expérience", "nuancee": "Nuancées (résultats dans les deux sens)",
                  "observee": "Observées (comparaisons entre concepts, pas encore testées)",
                  "hypothese": "Hypothèses à tester", "contredite": "Contredites par l'expérience (à ne pas suivre)"}
        lecons = sorted(self.lecons, key=lambda x: (ordre.get(x.statut, 9), x.id))[:n_lecons]
        for statut in ordre:
            groupe = [x for x in lecons if x.statut == statut]
            if not groupe:
                continue
            lignes.append(f"\n### {titres[statut]}")
            for x in groupe:
                t = f"- {x.id} : {x.enonce}"
                contre = sum(p.verdict == "contraire" for p in x.preuves if p.type in ("comparaison", "observation"))
                if contre and x.statut in ("hypothese", "observee"):
                    t += f" (Attention : {contre} constat(s) contre cette leçon dans le carnet ; à vérifier avant de s'en servir.)"
                if x.mecanisme and not compact:
                    t += f" Mécanisme : {x.mecanisme}"
                if x.portee and not compact:
                    t += f" Portée : {x.portee}"
                exp = [p for p in x.preuves if p.type == "experience"]
                if exp and not compact:
                    t += " Expériences : " + "; ".join(
                        f"{p.fiches[0]} -> {p.fiches[1]}, {p.grandeur} {_nombre(p.valeurs[0], 3)} -> "
                        f"{_nombre(p.valeurs[1], 3)} ({p.verdict.replace('_', ' ')})" for p in exp[-3:]) + "."
                lignes.append(t)
        meilleures = self.meilleures(2 if compact else 3)
        if meilleures:
            lignes.append("\n### Meilleurs concepts mesurés (même chaîne de calcul, valeurs optimisées)")
            for f in meilleures:
                lignes.append(f"- {f.resume()}")
                if f.point_critique.get("texte") and not compact:
                    lignes.append(f"  Point critique : {f.point_critique['texte']}")
                inutiles = [x for x in f.faces if x.get("part_peu_utile", 0) >= 0.5 and x.get("part_masse", 0) >= 0.05]
                if inutiles:
                    lignes.append("  Matière peu utile : " + ", ".join(
                        f"{x['id']} ({100 * x['part_masse']:.0f} % de la masse, {100 * x['part_peu_utile']:.0f} % "
                        "de sa surface sous 20 % de la contrainte utile)" for x in inutiles))
        if parent and self.a_fiche(parent):
            f = self.fiche(parent)
            lignes.append(f"\n### Concept de départ : {f.resume()}")
            lignes += [f"  Piste : {p}" for p in f.pistes[:4]]
            faites = [e for e in self.experiences if e.parent == parent and e.statut != "a_faire"]
            if faites:
                lignes.append("  Modifications déjà essayées sur ce concept (ne pas les refaire) :")
                lignes += [f"  - {e.texte()}" for e in faites[-(4 if compact else 10):]]
        a_faire = self.experiences_a_faire()
        if a_faire and not compact:
            lignes.append("\n### Expériences proposées, pas encore faites")
            lignes += [f"- {e.texte()}" for e in a_faire[:5]]
        calib = self.texte_calibration()
        if calib:
            lignes.append(f"\n### Prédictions physiques vérifiées par le calcul (tolérance {sp.TOLERANCE_PCT:g} % sur le kg/kN)")
            lignes += [f"- {t}" for t in (calib[:1] if compact else calib)]
            if not compact:
                lignes += [f"- {p.texte()}" for p in [p for p in self.predictions if p.statut == "verifiee"][-4:]]
        stats = self.texte_statistiques() if not compact else []
        if stats:
            lignes.append("\n### Statistiques (calculées par Gabarit sur toutes les fiches)")
            lignes += [f"- {t}" for t in stats]
        return "\n".join(lignes)

    def texte(self) -> str:
        """Le carnet en entier, pour un humain (markdown)."""
        lignes = ["# Carnet d'apprentissage de Gabarit", "",
                  (f"{len(self.fiches)} fiche(s), {len(self.lecons)} leçon(s), {len(self.experiences)} expérience(s)"
                   + (f", {len(self.predictions)} prédiction(s) physique(s)" if self.predictions else "") + "."), "",
                  "## Concepts mesurés", "",
                  "| Fiche | Concept | Faisable | kg/kN | Masse (kg) | Effort tenu (kN) | L_eq (mm) | eta | Flexion | Flambement | Limite | Parent |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for f in sorted(self.fiches, key=lambda f: (not f.faisable, valeur(f, "kg_par_kN") if math.isfinite(valeur(f, "kg_par_kN")) else 1e9)):
            m, d = f.mesures, f.decomposition
            lignes.append(f"| {f.id} | {f.nom} | {'oui' if f.faisable else 'non'} | {_nombre(m.get('kg_par_kN'), 4)} | "
                          f"{_nombre(m.get('masse_kg'), 3)} | {_nombre((m.get('effort_tenu_N') or math.nan) / 1000, 2)} | "
                          f"{_nombre(d.get('l_eq_mm'), 0)} | {_nombre(d.get('eta'), 3)} | "
                          f"{_nombre(100 * (d.get('part_flexion') or math.nan), 0)} % | {_nombre(m.get('flambement'), 2)} | "
                          f"{f.limite or '-'} | {f.parent or '-'} |")
        lignes += ["", "## Leçons", ""]
        for x in self.lecons:
            lignes.append(f"- **{x.id}** ({x.statut}, {x.source}) : {x.enonce}"
                          + (f" *Mécanisme* : {x.mecanisme}" if x.mecanisme else "")
                          + (f" *Portée* : {x.portee}" if x.portee else "") + f" *Preuves* : {x.bilan_preuves()}.")
            lignes += [f"  - {texte_preuve(p)}" for p in x.preuves]
        if self.experiences:
            lignes += ["", "## Expériences", ""] + [f"- {e.texte()}" for e in self.experiences]
        if self.predictions:
            lignes += ["", "## Prédictions physiques", "",
                       (f"Écrites avant le calcul ; juste = kg/kN à {sp.TOLERANCE_PCT:g} % près et bon mode de ruine "
                        "(le lieu du point critique, s'il est prédit, est jugé à part)."), ""]
            lignes += [f"- {p.texte()}" for p in self.predictions]
            calib = self.texte_calibration()
            if calib:
                lignes += ["", "Calibration des principes :", ""] + [f"- {t}" for t in calib]
        s = self.statistiques()
        lignes += ["", f"## Statistiques ({s['n']} concept(s) faisable(s))", ""]
        lignes += [f"- {t}" for t in self.texte_statistiques()]
        for lien in s.get("liens", []):
            lignes.append(f"  - {CARACTERISTIQUES[lien['caracteristique']]} : Spearman {lien['rho']:+.2f}, p "
                          f"{lien['p']:.3f}, p corrigé (Holm) {lien['p_holm']:.3f}")
        if s["n"] < 3:
            lignes.append("- Moins de trois concepts faisables : pas encore de part de L_eq et de eta dans les écarts.")
        if s["n"] < 5:
            lignes.append("- Moins de cinq concepts faisables : pas encore de recherche de liens avec les caractéristiques.")
        return "\n".join(lignes) + "\n"


def caracteristiques(concept) -> dict:
    """Caractéristiques comparables d'un concept à l'autre, tirées de sa géométrie."""
    from gabarit.concepts import evidements as ev

    geo = concept.geometrie()
    faces = list(geo.faces)
    plis, soudures, doublages = set(), set(), set()
    for f in faces:
        for nature, g, _ in ev.liaisons(f, faces):
            cle = tuple(sorted((f.id, g.id)))
            {"pli": plis, "soudure": soudures, "doublage": doublages}[nature].add(cle)
    definition = getattr(concept, "definition", {})
    ids = {f.id for f in faces}
    retours = 0
    for tole in definition.get("toles", []):
        for k, r in enumerate(tole.get("retours", [])):
            nom = r.get("nom", "retour_" + str(r.get("face")) + "_" + str(k))  # nom par défaut de format.py
            retours += f"{tole['nom']}.{nom}" in ids  # un retour de hauteur nulle n'est pas construit
    axes = np.eye(3)
    inclinees = sum(max(abs(float(np.dot(f.normale(), a))) for a in axes) < 0.999 for f in faces)
    aire = sum(f.aire_nette() for f in faces)
    return {
        "n_toles": len(definition.get("toles", [])) or len({f.tole for f in faces}),
        "n_faces": len(faces),
        "n_plis": len(plis),
        "n_soudures": len(soudures),
        "n_doublages": len(doublages),
        "n_retours": int(retours),
        "n_faces_inclinees": int(inclinees),
        "n_evidements": sum(len(f.trous) for f in faces),
        "aire_evidements_cm2": round(sum(f.aire_trous for f in faces) / 100, 1),
        "epaisseur_moy_mm": round(sum(f.t * f.aire_nette() for f in faces) / max(aire, 1e-9), 3),
        "aire_tole_cm2": round(aire / 100, 1),
    }


def _garder(cls, d: dict) -> dict:
    noms = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in noms}


def _nombre_ou_nan(x) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else math.nan


def _propre(x):
    """Nombre JSON (NaN et infini deviennent None)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def _nombre(x, n: int) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{x:.{n}f}" if math.isfinite(x) else "-"


def _signe(x) -> str:
    return f"{x:+.0f} %" if isinstance(x, (int, float)) and math.isfinite(x) else "-"


def _relatif(chemin) -> str:
    if not chemin:
        return ""
    p = Path(chemin).resolve()
    return p.relative_to(RACINE).as_posix() if p.is_relative_to(RACINE) else str(p)  # même écriture sous Windows
