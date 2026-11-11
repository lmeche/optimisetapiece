"""Calculs de prédimensionnement : la physique des tôles en formules courtes, pour prévoir avant de dessiner.

Unités : N, mm, MPa (N/mm²), kg. Chaque fonction dit son domaine de validité ; ce sont des ordres de grandeur
que le calcul par éléments finis de Gabarit tranche ensuite (voir gabarit/savoir/prediction.py).

Sources des formules :
- treillis à deux barres : statique ; la longueur équivalente L_eq = somme(|N| l) / F est celle du compte
  rendu (gabarit/analyse/compte_rendu.py), d'où kg/kN = 1000 rho / Re * L_eq / eta ;
- flambement d'Euler, voilement de plaque (k = 4 sur deux bords, 0,43 avec un bord libre) : résistance des
  matériaux classique ; limites de parois comprimées de classe 3 : EN 1993-1-1, tableau 5.2 (42 epsilon et
  14 epsilon) ;
- écrouissage au pliage : EN 1993-1-3, paragraphe 3.2.2 ;
- soudure d'angle, méthode simplifiée : EN 1993-1-8, paragraphe 4.5.3.3 ; vis en traction : tableau 3.4.
Les valeurs des normes sont reprises de mémoire : à recouper avec les textes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Rm minimal (EN 10025-2, tôles de 3 à 100 mm ; le S235 sous 3 mm est aussi à 360 MPa) : sert à l'écrouissage
# et aux soudures, le cahier des charges ne donnant que Re.
RM_MIN = {"S235": 360.0, "S275": 410.0, "S355": 470.0}
K_SOUDE = 0.6  # longueur de flambement / longueur d'une barre soudée aux deux bouts (L + V recalé : 0,58)
COEF_FLAMBEMENT_MIN = 3.0
BETA_W = {"S235": 0.8, "S275": 0.85, "S355": 0.9}
GAMMA_M2 = 1.25


def d_plaque(E: float, nu: float) -> float:
    """Raideur de plaque réduite pi² E / (12 (1 - nu²)), en MPa : sigma_cr = k D (t/b)²."""
    return math.pi ** 2 * E / (12 * (1 - nu ** 2))


def kg_par_kN(l_eq: float, eta: float, rho: float, Re: float) -> float:
    """Identité de Gabarit : kg/kN = 1000 rho / Re * L_eq / eta (rho en kg/m³, L_eq en mm)."""
    return 1000 * (rho * 1e-9) / Re * l_eq / eta


# ---------- chemin d'effort ----------
@dataclass
class Treillis:
    h: float
    L: float
    disposition: str
    tirant_N: float
    bielle_N: float
    l_tirant: float
    l_bielle: float
    l_eq: float

    def borne_kg_par_kN(self, rho: float, Re: float) -> float:
        """kg/kN d'un treillis où toute la matière travaille à Re (eta = 1) : la borne de ce chemin d'effort."""
        return kg_par_kN(self.l_eq, 1.0, rho, Re)


def treillis(F: float, L: float, h: float, disposition: str = "tirant_horizontal") -> Treillis:
    """Treillis à deux barres qui porte F à la distance L de la paroi, ses attaches écartées de h sur la paroi.

    - tirant_horizontal : tirant horizontal au niveau de la charge, bielle qui descend vers la paroi ;
      L_eq = (2 L² + h²) / h, minimale pour h = L racine(2) ;
    - symetrique : charge à mi-hauteur, tirant qui monte et bielle qui descend ;
      L_eq = (2 L² + h²/2) / h, 5 à 8 % de moins à h égal."""
    if h <= 0 or L <= 0:
        raise ValueError("h et L doivent être positifs")
    if disposition == "tirant_horizontal":
        lt, lb = L, math.hypot(L, h)
    elif disposition == "symetrique":
        lt = lb = math.hypot(L, h / 2)
    else:
        raise ValueError("disposition : tirant_horizontal ou symetrique")
    T, C = F * lt / h, F * lb / h
    return Treillis(h, L, disposition, T, C, lt, lb, (T * lt + C * lb) / F)


def hauteur_optimale(L: float, disposition: str = "tirant_horizontal") -> float:
    return L * math.sqrt(2) if disposition == "tirant_horizontal" else 2 * L


def contrainte_flexion_plaque(M: float, b: float, t: float) -> float:
    """Contrainte de flexion d'une tôle rectangulaire de largeur b et d'épaisseur t (M en N·mm) : 6 M / (b t²).
    Posée sur chant, la même tôle a b et t échangés : b/t fois plus résistante, (b/t)² fois plus rigide."""
    return 6 * M / (b * t * t)


# ---------- sections ----------
@dataclass
class Section:
    nom: str
    A: float  # mm²
    I_faible: float  # mm⁴, axe de flambement le plus défavorable
    I_fort: float
    parois: list[tuple[str, float, float, int]]  # (nom, largeur b, épaisseur t, bords tenus 1 ou 2)
    plaque: bool  # section plate : flambement en plaque (facteur 1/(1 - nu²))
    n_plis: int


def section_plaque(b: float, t: float) -> Section:
    return Section(f"plaque {b:g} x {t:g}", b * t, b * t ** 3 / 12, t * b ** 3 / 12, [], True, 0)


def section_u(ame: float, aile: float, t: float) -> Section:
    """U (âme et deux ailes de même épaisseur) mesuré sur les cotes extérieures, plis vifs. I_faible : flexion
    qui fait tourner l'âme autour d'un axe parallèle à elle (le flambement d'une plaque pliée de deux retours)."""
    a_ame, a_aile = ame * t, aile * t
    A = a_ame + 2 * a_aile
    yc = (a_ame * t / 2 + 2 * a_aile * aile / 2) / A
    I_faible = ame * t ** 3 / 12 + a_ame * (yc - t / 2) ** 2 + 2 * (t * aile ** 3 / 12 + a_aile * (aile / 2 - yc) ** 2)
    I_fort = t * ame ** 3 / 12 + 2 * (aile * t ** 3 / 12 + a_aile * (ame / 2) ** 2)
    return Section(f"U {ame:g} x {aile:g} x {t:g}", A, I_faible, I_fort,
                   [("âme", ame, t, 2), ("aile", aile, t, 1)], False, 2)


def section_depuis(d: dict) -> Section:
    """{"type": "plaque", "b": .., "t": ..} ou {"type": "U", "ame": .., "aile": .., "t": ..}."""
    ty = str(d.get("type", "")).lower()
    if ty == "plaque":
        return section_plaque(float(d["b"]), float(d["t"]))
    if ty in ("u", "profil_u", "omega"):
        return section_u(float(d["ame"]), float(d["aile"]), float(d["t"]))
    raise ValueError("section : type « plaque » (b, t) ou « U » (ame, aile, t)")


# ---------- instabilités ----------
def euler(E: float, I: float, l: float, K: float = 1.0) -> float:
    """Charge critique d'Euler (N) d'une barre de longueur l, longueur de flambement K l."""
    return math.pi ** 2 * E * I / (K * l) ** 2


def voilement(t: float, b: float, bords: int, E: float, nu: float) -> float:
    """Contrainte critique de voilement (MPa) d'une paroi comprimée de largeur b : k D (t/b)², k = 4 tenue sur
    ses deux bords, 0,43 avec un bord libre (bords simplement appuyés, sans imperfection)."""
    k = 4.0 if bords >= 2 else 0.43
    return k * d_plaque(E, nu) * (t / b) ** 2


def elancement_max(Re: float, bords: int, E: float = 210000.0, nu: float = 0.3) -> dict:
    """b/t maximal d'une paroi comprimée : élastique (voilement à Re) et Eurocode classe 3 (plus prudent)."""
    k = 4.0 if bords >= 2 else 0.43
    eps = math.sqrt(235.0 / Re)
    return {"elastique": math.sqrt(k * d_plaque(E, nu) / Re), "eurocode_classe_3": (42 if bords >= 2 else 14) * eps}


@dataclass
class Barre:
    nom: str
    section: Section
    longueur: float
    K: float
    effort_N: float
    contrainte: float
    euler_N: float
    facteur_euler: float
    voilement: list[tuple[str, float, float]]  # (paroi, sigma_cr, facteur)
    facteur_resistance: float
    masse_kg: float

    @property
    def facteur_flambement(self) -> float:
        return min([self.facteur_euler] + [f for _, _, f in self.voilement])

    def mode(self) -> str:
        modes = [("flambement d'ensemble", self.facteur_euler)] + [(f"voilement de l'{p}", f) for p, _, f in self.voilement]
        return min(modes, key=lambda m: m[1])[0]


def barre_comprimee(nom: str, section: Section, longueur: float, effort_N: float, E: float, nu: float, Re: float,
                    rho: float, K: float = K_SOUDE) -> Barre:
    """Bielle de section donnée sous l'effort de compression effort_N (charge de référence, 1 t ici)."""
    N = abs(effort_N)
    sigma = N / section.A
    ncr = euler(E, section.I_faible, longueur, K) / ((1 - nu ** 2) if section.plaque else 1.0)
    voil = [(p, voilement(t, b, bords, E, nu), voilement(t, b, bords, E, nu) / sigma) for p, b, t, bords in section.parois]
    return Barre(nom, section, longueur, K, N, sigma, ncr, ncr / N, voil, Re / sigma, section.A * longueur * rho * 1e-9)


# ---------- matériau ----------
def ecrouissage(fyb: float, fu: float, t: float, n_plis: float, A: float, k: float = 5.0) -> float:
    """Limite élastique moyenne d'une section pliée à froid (EN 1993-1-3, 3.2.2) : fyb + (fu - fyb) k n t² / A,
    plafonnée à (fu + fyb) / 2. k = 5 au pliage à la presse (7 au profilage). n : plis à 90° de rayon intérieur
    au plus 5 t. Seulement pour une section non réduite par le voilement, et perdu près d'une soudure."""
    return min(fyb + (fu - fyb) * k * n_plis * t * t / A, (fu + fyb) / 2)


def rm_min(nuance: str) -> float:
    return RM_MIN.get(nuance.upper().split()[0], 360.0)


# ---------- assemblages ----------
def fvw_d(fu: float, beta_w: float = 0.8, gamma_m2: float = GAMMA_M2) -> float:
    """Résistance de calcul au cisaillement d'une soudure d'angle, méthode simplifiée (MPa)."""
    return fu / (math.sqrt(3) * beta_w * gamma_m2)


def gorge_min(effort_N: float, longueur: float, fu: float, beta_w: float = 0.8, cordons: int = 2) -> float:
    """Gorge a (mm) qui reprend effort_N sur `cordons` cordons de `longueur` mm."""
    return abs(effort_N) / (fvw_d(fu, beta_w) * cordons * longueur)


def vis_traction_rd(fub: float = 800.0, As: float = 58.0, gamma_m2: float = GAMMA_M2) -> float:
    """Résistance de calcul en traction d'une vis (N) : 0,9 fub As / gamma_M2 (M10 8.8 : 33,4 kN)."""
    return 0.9 * fub * As / gamma_m2
