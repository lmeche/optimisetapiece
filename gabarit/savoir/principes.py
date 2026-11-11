"""Base de savoir physique : comment une pièce en tôle porte un effort, en principes vérifiables.

Chaque principe donne une règle de conception, son mécanisme, sa formule, son domaine de validité et la
grandeur que Gabarit mesure pour le vérifier. Les principes ne sont pas des exigences : le cahier des charges
dit ce qu'on exige, ce module dit comment on conçoit. Une IA qui s'en sert l'écrit dans sa prédiction (champ
« prediction » d'un concept, liste « principes ») ; le carnet compare ensuite la prédiction au calcul par
éléments finis et tient, pour chaque principe, le compte de ses prédictions justes (calibration).

Le raisonnement complet, chiffres du L + V à l'appui, est dans le document de réflexion « Gabarit : donner à
l'IA le sens physique des tôles ».
"""
from __future__ import annotations

from dataclasses import dataclass

from gabarit.cahier_des_charges import CahierDesCharges
from gabarit.savoir import calculs as ca


@dataclass(frozen=True)
class Principe:
    id: str
    titre: str
    regle: str
    mecanisme: str
    formule: str
    validite: str
    verifie: str


PRINCIPES = [
    Principe(
        "P1", "Un couple plutôt que de la flexion",
        "Porter le moment par un tirant tendu et une bielle comprimée écartés de la plus grande hauteur h possible.",
        "Le moment F.L devient deux efforts axiaux d'environ F.L/h : plus h est grand, moins il faut de matière.",
        "tirant horizontal : T = F.L/h, C = F.racine(L² + h²)/h, L_eq = (2L² + h²)/h, minimale pour h = L.racine(2)",
        "barres droites articulées ; la place est bornée par le socle (vis en haut, semelle en bas)",
        "L_eq mesurée face à la borne du treillis ; h réellement utilisé"),
    Principe(
        "P2", "Travail dans le plan",
        "Chaque tôle doit recevoir l'effort dans son plan (traction, compression, cisaillement), pas à plat.",
        "À plat, une tôle porte en flexion 6M/(b.t²) ; sur chant, b/t fois plus en résistance et (b/t)² fois plus "
        "en rigidité.",
        "sigma_flexion = 6M/(b.t²) ; sigma_membrane = N/(b.t)",
        "toujours ; la flexion locale reste inévitable là où l'effort entre",
        "part de l'énergie en flexion (part_flexion), globale et par face"),
    Principe(
        "P3", "Comprimé veut dire plié",
        "Toute paroi comprimée reste trapue (b/t petit) ou reçoit un pli, un retour ou un raidisseur.",
        "Une tôle mince comprimée flambe bien avant Re ; un pli multiplie l'inertie sans ajouter de matière et tient "
        "le bord de la paroi voisine.",
        "Euler : N_cr = pi².E.I/(K.l)², K proche de 0,6 pour une barre soudée aux deux bouts ; voilement : "
        "sigma_cr = k.189 800.(t/b)² MPa, k = 4 (deux bords tenus), 0,43 (un bord libre)",
        "plaques parfaites ; les imperfections de soudage et de pliage baissent la charge critique (d'où le facteur 3)",
        "facteur de flambement ; b/t de chaque paroi comprimée"),
    Principe(
        "P4", "Un appui sous la charge",
        "Placer une tôle dans son plan juste sous l'axe de l'effort, et pas de bord libre au ras du disque d'appui.",
        "Une tôle chargée à plat entre un appui et un bord libre plie localement : le pic de contrainte y fixe "
        "l'effort tenu de toute la pièce.",
        "flexion locale d'un porte-à-faux court : sigma proche de 6.F.a/(b.t²), a distance de l'appui à la charge",
        "zone d'introduction de l'effort ; la valeur exacte dépend du maillage, les tendances non",
        "contrainte et position du point critique ; distance entre l'appui et l'axe de l'effort"),
    Principe(
        "P5", "La compression passe par contact",
        "Appuyer la bielle sur la face du socle par contact, sous les vis ; seul le tirant passe par les vis.",
        "Le contact transmet la compression sans vis ni soudure ; les vis ne reprennent que la traction, qui fait "
        "levier si la platine fléchit.",
        "traction par vis proche de T/n, majorée par l'effet de levier ; M10 8.8 : 0,9.800.58/1,25 = 33,4 kN",
        "contact unilatéral, pris en compte dans le calcul de Gabarit",
        "traction maximale par vis"),
    Principe(
        "P6", "Pas de matière oisive",
        "Amincir une tôle peu chargée avant de la trouer ; garder la matière là où l'effort passe.",
        "eta, l'utilisation moyenne de la matière, pèse plus que la forme dans le kg/kN : une tôle à 10 % de la "
        "contrainte utile porte 10 fois son poids pour rien.",
        "kg/kN = 1000.rho/Re . L_eq/eta",
        "toujours ; une poche ne paie que si elle ne déplace pas le point critique (ESO du L + V : non)",
        "eta et part de surface peu utile, par face"),
    Principe(
        "P7", "L'épaisseur suit le mode de ruine",
        "Fin et plié pour ce qui travaille en traction-compression ; épais seulement là où la flexion locale est "
        "inévitable. Deux tôles permettent deux épaisseurs.",
        "La traction-compression croît en t, la flexion en t², le flambement de plaque en t³ : une tôle au maximum "
        "d'épaisseur signale une flexion ou un flambement à supprimer.",
        "masse en t ; résistance en flexion en t² ; voilement en t² (contrainte) et t³ (effort)",
        "épaisseurs de l'atelier ; aile minimale de pliage croissante avec t",
        "épaisseur de chaque tôle face au mode qui la limite"),
    Principe(
        "P8", "Des liaisons continues",
        "Souder sur toute la longueur des arêtes communes, pas de fin de soudure ni de trou dans une zone chargée.",
        "Une fin de soudure ou un trou concentre l'effort (Kt proche de 3 autour d'un trou dans une tôle tendue).",
        "soudure d'angle : f_vw,d = fu/(racine(3).beta_w.gamma_M2) = 208 MPa en S235 ; gorge a = N/(f_vw,d.longueur)",
        "statique ; la soudure est modélisée comme parfaite dans le calcul",
        "contrainte aux extrémités de jonction ; gorge nécessaire"),
    Principe(
        "P9", "Le matériau réel",
        "L'écrouissage n'aide que dans les plis, loin des soudures ; la réserve plastique n'est pas comptée avec le "
        "critère de première plastification.",
        "Le pliage à froid relève la limite élastique moyenne de la section (EN 1993-1-3) ; la chaleur d'une soudure "
        "l'efface. En flexion, une section garde 1,5 fois sa première plastification avant la ruine, en traction "
        "non.",
        "f_ya = f_yb + (f_u - f_yb).k.n.t²/A_g, au plus (f_u + f_yb)/2, k = 5 à la presse",
        "section non réduite par le voilement ; plis de rayon intérieur au plus 5t",
        "à calibrer quand le Re réel des tôles et la règle de l'essai seront connus"),
]
IDS = {p.id for p in PRINCIPES}


def principe(ident: str) -> Principe:
    return next(p for p in PRINCIPES if p.id == ident)


def hauteur_disponible(cdc: CahierDesCharges) -> dict:
    """Place verticale pour le couple : de la semelle (plus son dégagement) à la rangée de vis la plus haute."""
    from gabarit.concepts.format import DEGAGEMENT_SEMELLE

    z_vis = max(t.z for t in cdc.grille_fixation.trous)
    z_bas = cdc.encombrement.z_min + DEGAGEMENT_SEMELLE
    return {"z_bas_mm": z_bas, "z_vis_max_mm": z_vis, "h_mm": z_vis - z_bas}


def chiffres_du_cas(cdc: CahierDesCharges) -> list[str]:
    """Les ordres de grandeur du cas, calculés depuis le cahier des charges."""
    ch, m = cdc.chargements[0], cdc.materiau
    F, L = ch.force, ch.distance_encastrement
    hd = hauteur_disponible(cdc)
    t_dispo = ca.treillis(F, L, hd["h_mm"])
    deux, un = ca.elancement_max(m.Re, 2, m.E, m.nu), ca.elancement_max(m.Re, 1, m.E, m.nu)
    lignes = [
        f"Moment à la face du socle : {F:g} N x {L:g} mm = {F * L / 1e6:.2f} kN.m.",
        (f"Hauteur disponible pour le couple : de z = {hd['z_bas_mm']:g} (semelle et dégagement) à z = "
         f"{hd['z_vis_max_mm']:g} (vis les plus hautes), soit h = {hd['h_mm']:g} mm. Le dessus peut monter plus haut "
         "que les vis, mais la traction doit redescendre jusqu'à elles."),
        (f"Treillis à cette hauteur : tirant {t_dispo.tirant_N / 1000:.1f} kN, bielle {t_dispo.bielle_N / 1000:.1f} kN, "
         f"L_eq {t_dispo.l_eq:.0f} mm, borne {t_dispo.borne_kg_par_kN(m.rho, m.Re):.3f} kg/kN (toute la matière à Re)."),
        (f"Paroi comprimée en {m.nom} : b/t au plus {deux['eurocode_classe_3']:.0f} tenue sur deux bords, "
         f"{un['eurocode_classe_3']:.0f} avec un bord libre (Eurocode classe 3 ; {deux['elastique']:.0f} et "
         f"{un['elastique']:.0f} en élastique pur)."),
        f"Raideur de plaque : sigma_cr = k x {ca.d_plaque(m.E, m.nu):.0f} MPa x (t/b)².",
    ]
    return lignes


def pour_prompt(cdc: CahierDesCharges, compact: bool = False) -> str:
    """Section « Savoir physique » du prompt de l'IA Concepteur."""
    lignes = ["## Savoir physique (principes de Gabarit, à suivre et à citer dans ta prédiction)"]
    for p in PRINCIPES:
        t = f"- {p.id}, {p.titre} : {p.regle}"
        if not compact:
            t += f" Pourquoi : {p.mecanisme} Formule : {p.formule}."
        lignes.append(t)
    lignes.append("\nChiffres du cas :")
    lignes += [f"- {x}" for x in chiffres_du_cas(cdc)]
    lignes.append(
        "\nAvant d'écrire un concept, appelle l'outil predimensionner (treillis, bielles, parois, soudures) pour "
        "chiffrer ton idée, puis écris ta prédiction dans le champ « prediction » du concept : Gabarit la compare au "
        "calcul par éléments finis et en garde la trace pour chaque principe cité.")
    return "\n".join(lignes)


def texte() -> str:
    """Les principes en entier, pour un humain (markdown)."""
    lignes = ["# Savoir physique de Gabarit", ""]
    for p in PRINCIPES:
        lignes += [f"## {p.id}. {p.titre}", "", f"**Règle** : {p.regle}", "", f"**Mécanisme** : {p.mecanisme}", "",
                   f"**Formule** : {p.formule}", "", f"**Validité** : {p.validite}", "",
                   f"**Ce que Gabarit vérifie** : {p.verifie}", ""]
    return "\n".join(lignes)
