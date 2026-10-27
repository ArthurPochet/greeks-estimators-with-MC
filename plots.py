"""Graphiques des experiences (echelles log-log)."""
import os

import matplotlib
matplotlib.use("Agg")          # ecrit les images dans des fichiers
import matplotlib.pyplot as plt
import numpy as np

DOSSIER = "figures"
COULEURS = {"bump_independant": "tab:red", "bump_commun": "tab:blue",
            "pathwise": "tab:green", "likelihood_ratio": "tab:orange"}
# styles differents pour que deux courbes superposees restent visibles
STYLES = {"bump_independant": ("o", "-"), "bump_commun": ("o", "-"),
          "pathwise": ("s", "--"), "likelihood_ratio": ("^", "-")}


def _sauver(nom_fichier):
    os.makedirs(DOSSIER, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER, nom_fichier), dpi=150)
    plt.close()


def _courbe(x, y, methode):
    marqueur, trait = STYLES[methode]
    plt.loglog(x, y, marker=marqueur, linestyle=trait, label=methode,
               color=COULEURS[methode])


def tracer_erreur_vs_M(liste_M, erreurs, option_type="call"):
    """Une figure par Greek : erreur RMS en fonction de M.
    La droite pointillee a pour pente -1/2 (erreur proportionnelle a 1/racine(M))."""
    for nom in sorted({g for (_, g) in erreurs}):
        plt.figure(figsize=(6, 4.5))
        for (methode, greek), valeurs in erreurs.items():
            if greek == nom:
                _courbe(liste_M, valeurs, methode)
        M = np.array(liste_M)
        reference = min(v[0] for (_, g), v in erreurs.items() if g == nom)
        plt.loglog(M, reference * np.sqrt(M[0] / M), "k:", label="pente -1/2")
        plt.xlabel("Nombre de trajectoires M")
        plt.ylabel("Erreur RMS")
        plt.title("Erreur RMS du " + nom + " en fonction de M (" + option_type + ")")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        _sauver("erreur_vs_M_" + nom + "_" + option_type + ".png")


def tracer_erreur_vs_temps(temps, erreurs, option_type="call"):
    """Une figure par Greek : erreur RMS en fonction du temps de calcul.
    A temps egal, la courbe la plus basse est la meilleure methode."""
    for nom in sorted({g for (_, g) in erreurs}):
        plt.figure(figsize=(6, 4.5))
        for (methode, greek), valeurs in erreurs.items():
            if greek == nom:
                _courbe(temps[(methode, greek)], valeurs, methode)
        plt.xlabel("Temps de calcul du " + nom + " seul (s)")
        plt.ylabel("Erreur RMS")
        plt.title("Erreur du " + nom + " selon le temps de calcul (" + option_type + ")")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        _sauver("erreur_vs_temps_" + nom + "_" + option_type + ".png")


def tracer_erreur_vs_h(res, option_type="call", pas_choisis=None):
    """Une figure par Greek : erreur RMS du bump en fonction du pas h.
    L'etoile marque le pas retenu par la regle de choix (pas_choisis[(methode, greek)]) ;
    sans pas_choisis, le pas qui realise le minimum exact."""
    for nom in GREEKS_DES_BUMPS:
        plt.figure(figsize=(6, 4.5))
        for methode in ["bump_independant", "bump_commun"]:
            grille, valeurs = res[(methode, nom)]
            _courbe(grille, valeurs, methode)
            if pas_choisis is not None:
                i = list(grille).index(pas_choisis[(methode, nom)])
            else:
                i = int(np.argmin(valeurs))
            plt.plot(grille[i], valeurs[i], "*", markersize=16,
                     color=COULEURS[methode], markeredgecolor="black")
        plt.xlabel("Pas h")
        plt.ylabel("Erreur RMS du " + nom)
        plt.title("Bump : erreur du " + nom + " en fonction de h (" + option_type + ")")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        _sauver("erreur_vs_h_" + nom + "_" + option_type + ".png")


GREEKS_DES_BUMPS = ["delta", "gamma", "vega", "theta", "rho"]


def tracer_temps(liste_M, temps, option_type="call", greek="delta"):
    """Temps de calcul (median) d'un Greek seul en fonction de M. Le delta sert de
    reference : les quatre methodes le donnent. Les temps des autres Greeks sont
    dans resultats/temps_<option>.csv."""
    plt.figure(figsize=(6, 4.5))
    for (methode, g), valeurs in temps.items():
        if g == greek:
            _courbe(liste_M, valeurs, methode)
    plt.xlabel("Nombre de trajectoires M")
    plt.ylabel("Temps du " + greek + " seul (s)")
    plt.title("Temps de calcul du " + greek + " (" + option_type + ")")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    _sauver("temps_" + option_type + ".png")


def tracer_h_vs_M(liste_M, pas, option_type="call"):
    """Pas h retenu en fonction de M, une figure par methode de bump, une courbe par Greek.
    pas[(methode, greek, M)] = h. Pour le bump independant, la theorie prevoit h* en M^(-1/6)."""
    for methode in ["bump_independant", "bump_commun"]:
        plt.figure(figsize=(6, 4.5))
        for greek in GREEKS_DES_BUMPS:
            plt.loglog(liste_M, [pas[(methode, greek, M)] for M in liste_M],
                       marker="o", label=greek)
        plt.xlabel("Nombre de trajectoires M")
        plt.ylabel("Pas h retenu")
        plt.title("Pas h retenu selon M : " + methode + " (" + option_type + ")")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        _sauver("h_vs_M_" + methode + "_" + option_type + ".png")


def tracer_lr(res, option_type, balayage, titre_x):
    """Erreur relative du pathwise et du likelihood ratio, en fonction de la
    variable du balayage (sigma*racine(T) ou strike K). balayage = 'sigma' ou 'strike'."""
    for nom in ["delta", "gamma", "vega", "rho"]:
        plt.figure(figsize=(6, 4.5))
        for methode in ["pathwise", "likelihood_ratio"]:
            if (methode, nom) not in res:
                continue
            points = res[(methode, nom)]
            x = [a for a, _ in points]
            y = [b for _, b in points]
            marqueur, trait = STYLES[methode]
            tracer = plt.loglog if balayage in ("sigma", "T") else plt.semilogy
            tracer(x, y, marker=marqueur, linestyle=trait, label=methode,
                   color=COULEURS[methode])
        plt.xlabel(titre_x)
        plt.ylabel("Erreur RMS relative")
        plt.title("Erreur relative du " + nom + " (" + option_type + ")")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        _sauver("lr_" + balayage + "_" + nom + "_" + option_type + ".png")