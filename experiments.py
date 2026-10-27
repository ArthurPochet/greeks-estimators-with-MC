"""Experiences : comparaison de la precision des estimateurs de Greeks.

  1. Optimisation du pas h des bumps, refaite pour chaque M (call et put)
  2. Erreur en fonction de M, temps de calcul PAR GREEK, efficacite (call et put)
  3. Likelihood ratio : sensibilite a sigma, T et au strike K

Pour lancer :
    python experiments.py            # tout (compter quelques minutes)
    python experiments.py h          # seulement l'optimisation de h
    python experiments.py M          # seulement l'erreur en fonction de M
    python experiments.py lr         # seulement le test du likelihood ratio
L'experience 2 utilise, pour chaque M, les pas h trouves par l'experience 1
(fichier resultats/h_optimaux_<option>.csv ; sinon, les pas par defaut).
Il faut donc lancer "h" avant "M" (c'est l'ordre de "tout").
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
LISTE_M = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
M_FIGURE_H = 100_000            # M pour lequel on dessine l'erreur en fonction de h
# Regle de choix de h (voir choisir_pas) : l'erreur RMS d'une estimation sur 30 graines
# a une incertitude d'environ 1/racine(2*30) = 13 % ; 25 % correspond a deux ecarts-types.
TOLERANCE_PLATEAU = 0.25
# Greeks que chaque methode sait estimer (le bump donne les cinq).
GREEKS_PAR_METHODE = {
    "bump_independant": GREEKS,
    "bump_commun": GREEKS,
    "pathwise": ["delta", "vega", "rho"],
    "likelihood_ratio": ["delta", "gamma", "vega", "rho"],
}
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


def estimer(methode, M, rng, option_type="call", p=PARAMS, greeks=None, **pas):
    """UNE estimation des Greeks avec la methode demandee.
    Renvoie {nom du Greek: valeur estimee}. Les pas h sont passes en **pas.
    greeks : liste des Greeks voulus (None = tous ceux de la methode). Demander un
    seul Greek permet de mesurer son cout seul."""
    S, K, r, T, sigma = p["S"], p["K"], p["r"], p["T"], p["sigma"]
    if methode == "bump_independant":
        return bump_independant(S, K, r, T, sigma, option_type, M, rng,
                                greeks=greeks, **pas)
    if methode == "bump_commun":
        return bump_commun(S, K, r, T, sigma, option_type, M, rng,
                           greeks=greeks, **pas)
    Z = rng.standard_normal(M)
    if methode == "pathwise":
        return pathwise_greeks(S, K, r, T, sigma, option_type, Z, greeks=greeks)
    if methode == "likelihood_ratio":
        return likelihood_ratio_greeks(S, K, r, T, sigma, option_type, Z, greeks=greeks)
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
    """Erreur RMS de chaque Greek par bump en fonction de son pas h, pour UN M.

    Les graines (100 a 129) different de celles de l'experience 2 (0 a 29),
    pour ne pas choisir h sur le bruit des memes tirages.
    Pour chaque pas on ne calcule que les prix necessaires aux Greeks concernes.
    Retourne res[(methode, greek)] = (liste_h, liste_rmse)."""
    exact = greeks_exacts(option_type)
    res = {}
    for methode in ["bump_independant", "bump_commun"]:
        for param, (nom_pas, grille, greeks) in PAS.items():
            ecarts = {(g, h): [] for g in greeks for h in grille}
            for h in grille:
                for k in range(n_graines):
                    rng = np.random.default_rng(graine_depart + k)
                    estime = estimer(methode, M, rng, option_type, greeks=greeks,
                                     **{nom_pas: h})
                    for g in greeks:
                        ecarts[(g, h)].append(estime[g] - exact[g])
            for g in greeks:
                res[(methode, g)] = (list(grille), [rmse(ecarts[(g, h)]) for h in grille])
    return res


def choisir_pas(grille, valeurs, tolerance=TOLERANCE_PLATEAU):
    """Regle de choix de h : le PLUS PETIT h dont l'erreur RMS est a moins de
    `tolerance` (25 %) du minimum.

    Pourquoi : sur un plateau (cas des nombres communs), tous les h se valent a
    l'incertitude de mesure pres ; le minimum exact est alors le fruit du hasard et
    tombe souvent du cote des grands h, ou le biais (en h^2) est le plus fort. En
    prenant le plus petit h du plateau, on garde la meme precision avec un biais
    minimal. Quand la courbe a un vrai creux (bump independant, gamma), cette regle
    retombe sur le creux, car les pas voisins sont bien plus mauvais.
    Retourne (h retenu, erreur RMS en ce h, h qui realise le minimum exact)."""
    minimum = min(valeurs)
    h_minimum = grille[int(np.argmin(valeurs))]
    for h, v in sorted(zip(grille, valeurs)):
        if v <= (1 + tolerance) * minimum:
            return h, v, h_minimum


def experience_h_par_M(option_type="call", liste_M=LISTE_M, n_graines=30):
    """Refait l'optimisation de h pour CHAQUE M de la liste (le h optimal depend de M :
    plus M est grand, plus le bruit est faible, donc plus on peut prendre h petit pour
    reduire le biais ; pour le bump independant, h* varie comme M^(-1/6)).

    Retourne res_par_M[M] = res (voir experience_h)
             choix[(methode, greek, M)] = (h retenu, rmse, h du minimum exact)."""
    res_par_M = {}
    choix = {}
    for M in liste_M:
        res_par_M[M] = experience_h(option_type, M=M, n_graines=n_graines)
        for (methode, greek), (grille, valeurs) in res_par_M[M].items():
            choix[(methode, greek, M)] = choisir_pas(grille, valeurs)
        print("  optimisation de h terminee pour M =", M, "-", option_type)
    return res_par_M, choix


def ecrire_pas(option_type, res_par_M, choix):
    """Ecrit h_optimaux_<option>.csv (le pas retenu, pour chaque methode, Greek et M)
    et h_courbes_<option>.csv (toutes les courbes erreur-h, pour les refaire sans relancer)."""
    ecrire_csv("h_optimaux_" + option_type + ".csv",
               ["methode", "greek", "M", "h_optimal", "h_argmin", "rmse_au_choix"],
               [[m, g, M, h, h_min, v] for (m, g, M), (h, v, h_min) in sorted(choix.items())])
    lignes = []
    for M, res in res_par_M.items():
        for (methode, greek), (grille, valeurs) in sorted(res.items()):
            lignes += [[methode, greek, M, h, v] for h, v in zip(grille, valeurs)]
    ecrire_csv("h_courbes_" + option_type + ".csv",
               ["methode", "greek", "M", "h", "rmse"], lignes)


# ------------------------- experience 2 : erreur vs M, temps, efficacite
def charger_pas_optimaux(option_type):
    """Lit resultats/h_optimaux_<option>.csv.
    Retourne {(methode, greek, M): h}, ou None si le fichier n'existe pas ou date
    de l'ancien format (sans colonne M) : il faut alors relancer `python experiments.py h`."""
    chemin = os.path.join(DOSSIER, "h_optimaux_" + option_type + ".csv")
    if not os.path.exists(chemin):
        return None
    pas = {}
    with open(chemin) as f:
        for ligne in csv.DictReader(f):
            if "M" not in ligne:
                return None
            pas[(ligne["methode"], ligne["greek"], int(ligne["M"]))] = float(ligne["h_optimal"])
    return pas


def experience_M(liste_M, option_type="call", n_graines=30, pas_opt=None):
    """Erreur RMS, biais et temps de chaque (methode, Greek) pour chaque M.

    Chaque Greek est estime SEUL, avec son propre pas h (pour M donne) si c'est un
    bump : delta, vega, rho et theta coutent deux prix, le gamma trois. Le temps est
    donc celui de ce Greek seul (tirage de Z compris), median sur les graines.
    Les graines (0 a 29) different de celles de l'experience 1.

    Retourne erreurs[(methode, greek)] = liste des RMSE (une par M)
             biais[(methode, greek)]   = liste des erreurs moyennes signees
             temps[(methode, greek)]   = liste des temps medians (s)."""
    exact = greeks_exacts(option_type)
    erreurs, biais, temps = {}, {}, {}
    for methode in METHODES:
        est_bump = methode.startswith("bump")
        for M in liste_M:
            for g in GREEKS_PAR_METHODE[methode]:
                pas = {}
                if est_bump and pas_opt is not None:
                    pas = {NOM_PAS_DU_GREEK[g]: pas_opt[(methode, g, M)]}
                ecarts, durees = [], []
                for graine in range(n_graines):
                    rng = np.random.default_rng(graine)
                    debut = time.perf_counter()
                    estime = estimer(methode, M, rng, option_type, greeks=[g], **pas)
                    durees.append(time.perf_counter() - debut)
                    ecarts.append(estime[g] - exact[g])
                erreurs.setdefault((methode, g), []).append(rmse(ecarts))
                biais.setdefault((methode, g), []).append(float(np.mean(ecarts)))
                temps.setdefault((methode, g), []).append(float(np.median(durees)))
        print("  erreur en fonction de M terminee pour", methode, "-", option_type)
    return erreurs, biais, temps


def efficacite(erreurs, temps, nom):
    """(erreur RMS)^2 x temps du Greek seul, pour le plus grand M. Plus petit = meilleur."""
    return {m: erreurs[(m, nom)][-1] ** 2 * temps[(m, nom)][-1]
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

    for option_type in ["call", "put"]:
        print("\n=== Option :", option_type, "===")

        if quoi in ("tout", "h"):
            print("Experience 1 : optimisation de h pour chaque M ...")
            res_par_M, choix = experience_h_par_M(option_type, LISTE_M)
            ecrire_pas(option_type, res_par_M, choix)
            for M in (M_FIGURE_H, LISTE_M[-1]):
                print("\nPas retenus (plus petit h a moins de 25 % du minimum), M =", M,
                      "- option", option_type)
                for (m, g, M_), (h, v, h_min) in sorted(choix.items()):
                    if M_ == M:
                        print(f"   {m:17s} {g:6s} h = {h:<8g} (minimum exact : {h_min:<8g})"
                              f" RMSE = {v:.3e}")
            plots.tracer_erreur_vs_h(
                res_par_M[M_FIGURE_H], option_type,
                {(m, g): h for (m, g, M_), (h, _, _) in choix.items() if M_ == M_FIGURE_H})
            plots.tracer_h_vs_M(LISTE_M, {k: v[0] for k, v in choix.items()}, option_type)

        if quoi in ("tout", "M"):
            pas_opt = charger_pas_optimaux(option_type)
            if pas_opt is None:
                print("\nATTENTION : pas de h_optimaux_" + option_type + ".csv utilisable "
                      "(lancer d'abord `python experiments.py h`) : pas h par defaut.")
            print("\nExperience 2 : erreur en fonction de M"
                  + (" (pas h optimises pour chaque M)" if pas_opt else " (pas h par defaut)"),
                  "...")
            erreurs, biais, temps = experience_M(LISTE_M, option_type, pas_opt=pas_opt)
            print("\nTemps d'UN Greek seul (s), M =", LISTE_M[-1])
            for g in GREEKS:
                print("  ", g, "  ".join(f"{m}: {temps[(m, g)][-1]:.4f}"
                                        for m in METHODES if (m, g) in temps))
            lignes = []
            for g in GREEKS:
                eff = efficacite(erreurs, temps, g)
                print("\nEfficacite (RMSE^2 x temps, plus petit = mieux) :", g)
                for m, v in sorted(eff.items(), key=lambda x: x[1]):
                    print(f"   {m:17s} {v:.3e}")
                    lignes.append([g, m, v])
            entete_M = ["M=%d" % M for M in LISTE_M]
            ecrire_csv("efficacite_" + option_type + ".csv",
                       ["greek", "methode", "efficacite"], lignes)
            for nom, donnees in [("erreurs", erreurs), ("biais", biais), ("temps", temps)]:
                ecrire_csv(nom + "_" + option_type + ".csv", ["methode", "greek"] + entete_M,
                           [[m, g] + v for (m, g), v in sorted(donnees.items())])
            plots.tracer_erreur_vs_M(LISTE_M, erreurs, option_type)
            plots.tracer_erreur_vs_temps(temps, erreurs, option_type)
            plots.tracer_temps(LISTE_M, temps, option_type)

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
