"""Experiences : comparaison de la precision des estimateurs de Greeks.

  1. Optimisation du pas h des bumps (call et put)
  2. Erreur en fonction de M, temps de calcul, efficacite (call et put)
  3. Likelihood ratio : sensibilite a sigma*racine(T)

Pour lancer :
    python experiments.py            # tout (compter quelques minutes)
    python experiments.py h          # seulement l'optimisation de h
    python experiments.py M          # seulement l'erreur en fonction de M
    python experiments.py lr         # seulement le test du likelihood ratio
Les experiences 2 utilisent les pas h optimises trouves par l'experience 1
(fichier resultats/h_optimaux_<option>.csv ; sinon, les pas par defaut).
"""
import csv
import os
import sys
import time

import numpy as np

import black_scholes as bs
from estimators import (bump_commun, bump_independant, likelihood_ratio_greeks,
                        pathwise_greeks)

PARAMS = {"S": 100, "K": 100, "r": 0.05, "T": 1, "sigma": 0.2}
METHODES = ["bump_independant", "bump_commun", "pathwise", "likelihood_ratio"]
GREEKS = ["delta", "gamma", "vega", "theta", "rho"]
DOSSIER = "resultats"

# Grilles de pas h, par parametre que l'on decale.
# parametre : (nom de l'argument de bump_greeks, grille, Greeks obtenus)
PAS = {
    "S":     ("h_S",     [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], ["delta", "gamma"]),
    "sigma": ("h_sigma", [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.15],    ["vega"]),
    "r":     ("h_r",     [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3], ["rho"]),
    "T":     ("h_T",     [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 0.9],   ["theta"]),
}
PAS_DEFAUT = {"h_S": 1.0, "h_sigma": 0.01, "h_r": 0.001, "h_T": 0.01}
NOM_PAS_DU_GREEK = {"delta": "h_S", "gamma": "h_S", "vega": "h_sigma",
                    "rho": "h_r", "theta": "h_T"}


# ---------------------------------------------------------------- utilitaires
def greeks_exacts(option_type="call", p=PARAMS):
    """Valeurs exactes de Black-Scholes (la reference)."""
    args = (p["S"], p["K"], p["r"], p["T"], p["sigma"])
    if option_type == "call":
        return {"delta": bs.call_delta(*args), "gamma": bs.gamma(*args),
                "vega": bs.vega(*args), "theta": bs.call_theta(*args),
                "rho": bs.call_rho(*args)}
    return {"delta": bs.put_delta(*args), "gamma": bs.gamma(*args),
            "vega": bs.vega(*args), "theta": bs.put_theta(*args),
            "rho": bs.put_rho(*args)}


def estimer(methode, M, rng, option_type="call", p=PARAMS, **pas):
    """UNE estimation des Greeks avec la methode demandee.
    Renvoie {nom du Greek: valeur estimee}. Les pas h sont passes en **pas."""
    S, K, r, T, sigma = p["S"], p["K"], p["r"], p["T"], p["sigma"]
    if methode == "bump_independant":
        return bump_independant(S, K, r, T, sigma, option_type, M, rng, **pas)
    if methode == "bump_commun":
        return bump_commun(S, K, r, T, sigma, option_type, M, rng, **pas)
    Z = rng.standard_normal(M)
    if methode == "pathwise":
        return pathwise_greeks(S, K, r, T, sigma, option_type, Z)
    if methode == "likelihood_ratio":
        return likelihood_ratio_greeks(S, K, r, T, sigma, option_type, Z)
    raise ValueError("Methode inconnue : " + methode)


def rmse(ecarts):
    return float(np.sqrt(np.mean(np.array(ecarts) ** 2)))


def ecrire_csv(nom_fichier, entete, lignes):
    os.makedirs(DOSSIER, exist_ok=True)
    with open(os.path.join(DOSSIER, nom_fichier), "w", newline="") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(entete)
        ecrivain.writerows(lignes)


# ------------------------------------------- experience 1 : optimisation de h
def experience_h(option_type="call", M=100_000, n_graines=30, graine_depart=100):
    """Erreur RMS de chaque Greek par bump en fonction de son pas h.

    Les graines (100 a 129) different de celles de l'experience 2 (0 a 29),
    pour ne pas choisir h sur le bruit des memes tirages.
    Retourne res[(methode, greek)] = (liste_h, liste_rmse)."""
    exact = greeks_exacts(option_type)
    res = {}
    for methode in ["bump_independant", "bump_commun"]:
        for param, (nom_pas, grille, greeks) in PAS.items():
            ecarts = {(g, h): [] for g in greeks for h in grille}
            for h in grille:
                for k in range(n_graines):
                    rng = np.random.default_rng(graine_depart + k)
                    estime = estimer(methode, M, rng, option_type, **{nom_pas: h})
                    for g in greeks:
                        ecarts[(g, h)].append(estime[g] - exact[g])
            for g in greeks:
                res[(methode, g)] = (list(grille), [rmse(ecarts[(g, h)]) for h in grille])
        print("  optimisation de h terminee pour", methode, "-", option_type)
    return res


def pas_optimaux(res):
    """h qui minimise l'erreur RMS, pour chaque (methode, greek)."""
    optimaux = {}
    for (methode, greek), (grille, valeurs) in res.items():
        i = int(np.argmin(valeurs))
        optimaux[(methode, greek)] = (grille[i], valeurs[i])
    return optimaux


# ------------------------- experience 2 : erreur vs M, temps, efficacite
def charger_pas_optimaux(option_type):
    """Lit resultats/h_optimaux_<option>.csv ; None si le fichier n'existe pas."""
    chemin = os.path.join(DOSSIER, "h_optimaux_" + option_type + ".csv")
    if not os.path.exists(chemin):
        return None
    pas = {}
    with open(chemin) as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            pas[(ligne["methode"], ligne["greek"])] = float(ligne["h_optimal"])
    return pas


def experience_M(liste_M, option_type="call", n_graines=30, pas_opt=None):
    """Erreur RMS de chaque (methode, Greek) pour chaque M, et temps moyen.

    Pour les bumps, si pas_opt est donne, chaque Greek est estime avec son
    propre pas optimal (un appel de bump par pas different). Le temps mesure est
    celui d'UN appel, c'est-a-dire le cout pour obtenir les cinq Greeks.

    Retourne erreurs[(methode, greek)] = liste des RMSE (une par M)
             temps[methode]            = liste des temps moyens (s)."""
    exact = greeks_exacts(option_type)
    erreurs = {}
    temps = {m: [] for m in METHODES}
    for methode in METHODES:
        est_bump = methode.startswith("bump")
        for M in liste_M:
            ecarts = {}
            duree = 0.0
            for graine in range(n_graines):
                rng = np.random.default_rng(graine)
                debut = time.perf_counter()
                estime = estimer(methode, M, rng, option_type)
                duree += time.perf_counter() - debut
                if est_bump and pas_opt is not None:
                    # on remplace chaque Greek par son estimation avec h optimal
                    estime = {}
                    for g in GREEKS:
                        h = pas_opt[(methode, g)]
                        estime[g] = estimer(methode, M, rng, option_type,
                                            **{NOM_PAS_DU_GREEK[g]: h})[g]
                for nom, valeur in estime.items():
                    ecarts.setdefault(nom, []).append(valeur - exact[nom])
            temps[methode].append(duree / n_graines)
            for nom, liste in ecarts.items():
                erreurs.setdefault((methode, nom), []).append(rmse(liste))
        print("  erreur en fonction de M terminee pour", methode, "-", option_type)
    return erreurs, temps


def efficacite(erreurs, temps, nom):
    """(erreur RMS)^2 x temps pour le plus grand M. Plus petit = meilleur."""
    return {m: erreurs[(m, nom)][-1] ** 2 * temps[m][-1]
            for m in METHODES if (m, nom) in erreurs}


# --------------------------- experience 3 : le likelihood ratio selon sigma*racine(T) et K
def _erreurs_relatives(option_type, p, M, n_graines):
    """Erreur RMS relative (RMSE / |valeur exacte|) du pathwise et du likelihood ratio."""
    exact = greeks_exacts(option_type, p)
    sortie = {}
    for methode in ["pathwise", "likelihood_ratio"]:
        ecarts = {}
        for graine in range(n_graines):
            rng = np.random.default_rng(graine)
            estime = estimer(methode, M, rng, option_type, p)
            for nom, valeur in estime.items():
                ecarts.setdefault(nom, []).append(valeur - exact[nom])
        for nom, liste in ecarts.items():
            sortie[(methode, nom)] = rmse(liste) / abs(exact[nom])
    return sortie


def experience_lr(option_type="call", M=100_000, n_graines=30):
    """Le likelihood ratio se degrade-t-il quand sigma ou T est petit, ou quand
    l'option est loin de la monnaie ?

    Trois balayages, avec M fixe, erreur RELATIVE du pathwise et du likelihood ratio :
      - sigma varie (T = 1), a la monnaie avec r = 0
      - T varie (sigma = 0,2), a la monnaie avec r = 0
      - le strike K varie (parametres par defaut)
    r = 0 pour les deux premiers : le forward vaut alors S, donc K = S est
    vraiment a la monnaie et seul sigma ou T change.
    Retourne (res_sigma, res_T, res_K), chacun res[(methode, greek)] = liste de (x, erreur)."""
    res_sigma, res_T, res_K = {}, {}, {}
    p0 = dict(PARAMS, r=0.0)

    for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
        p = dict(p0, sigma=sigma)
        for cle, e in _erreurs_relatives(option_type, p, M, n_graines).items():
            res_sigma.setdefault(cle, []).append((sigma, e))

    for T in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        p = dict(p0, T=T)
        for cle, e in _erreurs_relatives(option_type, p, M, n_graines).items():
            res_T.setdefault(cle, []).append((T, e))

    for K in [70, 80, 90, 100, 110, 120, 130]:
        p = dict(PARAMS, K=K)
        for cle, e in _erreurs_relatives(option_type, p, M, n_graines).items():
            res_K.setdefault(cle, []).append((K, e))

    for res in (res_sigma, res_T, res_K):
        for cle in res:
            res[cle].sort()
    print("  test du likelihood ratio termine -", option_type)
    return res_sigma, res_T, res_K


# ---------------------------------------------------------------- programme
if __name__ == "__main__":
    import plots

    quoi = sys.argv[1] if len(sys.argv) > 1 else "tout"
    liste_M = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]

    for option_type in ["call", "put"]:
        print("\n=== Option :", option_type, "===")

        if quoi in ("tout", "h"):
            print("Experience 1 : optimisation de h ...")
            res_h = experience_h(option_type)
            optimaux = pas_optimaux(res_h)
            ecrire_csv("h_optimaux_" + option_type + ".csv",
                       ["methode", "greek", "h_optimal", "rmse_au_minimum"],
                       [[m, g, h, v] for (m, g), (h, v) in sorted(optimaux.items())])
            print("\nPas optimaux (M = 100 000), option", option_type)
            for (m, g), (h, v) in sorted(optimaux.items()):
                print(f"   {m:17s} {g:6s} h = {h:<8g} RMSE = {v:.3e}")
            plots.tracer_erreur_vs_h(res_h, option_type)

        if quoi in ("tout", "M"):
            pas_opt = charger_pas_optimaux(option_type)
            print("\nExperience 2 : erreur en fonction de M"
                  + (" (pas h optimises)" if pas_opt else " (pas h par defaut)"), "...")
            erreurs, temps = experience_M(liste_M, option_type, pas_opt=pas_opt)
            print("\nTemps moyen par estimation (s), M =", liste_M[-1])
            for m in METHODES:
                print(f"   {m:17s} {temps[m][-1]:.4f}")
            lignes = []
            for g in GREEKS:
                eff = efficacite(erreurs, temps, g)
                print("\nEfficacite (RMSE^2 x temps, plus petit = mieux) :", g)
                for m, v in sorted(eff.items(), key=lambda x: x[1]):
                    print(f"   {m:17s} {v:.3e}")
                    lignes.append([g, m, v])
            ecrire_csv("efficacite_" + option_type + ".csv",
                       ["greek", "methode", "efficacite"], lignes)
            ecrire_csv("erreurs_" + option_type + ".csv",
                       ["methode", "greek"] + ["M=%d" % M for M in liste_M],
                       [[m, g] + v for (m, g), v in sorted(erreurs.items())])
            plots.tracer_erreur_vs_M(liste_M, erreurs, option_type)
            plots.tracer_erreur_vs_temps(temps, erreurs, option_type)
            plots.tracer_temps(liste_M, temps, option_type)

        if quoi in ("tout", "lr"):
            print("\nExperience 3 : likelihood ratio selon sigma, T et K ...")
            res_sigma, res_T, res_K = experience_lr(option_type)
            plots.tracer_lr(res_sigma, option_type, "sigma", "volatilite sigma (T = 1, r = 0)")
            plots.tracer_lr(res_T, option_type, "T", "maturite T (sigma = 0,2, r = 0)")
            plots.tracer_lr(res_K, option_type, "strike", "strike K")
            lignes = []
            for nom_balayage, res in [("sigma", res_sigma), ("T", res_T), ("strike", res_K)]:
                for (m, g), liste in sorted(res.items()):
                    lignes += [[nom_balayage, m, g, x, e] for x, e in liste]
            ecrire_csv("lr_" + option_type + ".csv",
                       ["balayage", "methode", "greek", "x", "erreur_relative"], lignes)

    print("\nFigures dans figures/, tableaux dans resultats/")