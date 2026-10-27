"""Tests des nouveautes : calcul d'un seul Greek, regle de choix de h."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

import estimators
from estimators import (bump_commun, bump_independant, likelihood_ratio_greeks,
                        pathwise_greeks)
from experiments import choisir_pas

S, K, r, T, sigma = 100, 100, 0.05, 1, 0.2
M = 50_000


# ----------------------------------------------------- un seul Greek a la fois
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bump_commun_un_seul_greek_donne_la_meme_valeur(option_type):
    """Avec la meme graine, demander un seul Greek donne exactement la valeur
    obtenue en demandant les cinq (le Z commun est tire en premier dans les deux cas)."""
    complet = bump_commun(S, K, r, T, sigma, option_type, M, np.random.default_rng(7))
    for greek in complet:
        seul = bump_commun(S, K, r, T, sigma, option_type, M,
                           np.random.default_rng(7), greeks=[greek])
        assert list(seul) == [greek]
        assert seul[greek] == complet[greek]


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_pathwise_et_lr_un_seul_greek(option_type):
    Z = np.random.default_rng(1).standard_normal(M)
    for fonction in (pathwise_greeks, likelihood_ratio_greeks):
        complet = fonction(S, K, r, T, sigma, option_type, Z)
        for greek in complet:
            seul = fonction(S, K, r, T, sigma, option_type, Z, greeks=[greek])
            assert list(seul) == [greek]
            assert seul[greek] == complet[greek]


def test_greek_non_fourni_leve_une_erreur():
    Z = np.random.default_rng(1).standard_normal(100)
    with pytest.raises(ValueError):
        pathwise_greeks(S, K, r, T, sigma, "call", Z, greeks=["gamma"])
    with pytest.raises(ValueError):
        likelihood_ratio_greeks(S, K, r, T, sigma, "call", Z, greeks=["theta"])
    with pytest.raises(ValueError):
        bump_commun(S, K, r, T, sigma, "call", 100, np.random.default_rng(0), greeks=["vol"])


@pytest.mark.parametrize("greeks, nb_prix", [
    (["delta"], 2), (["vega"], 2), (["rho"], 2), (["theta"], 2),
    (["gamma"], 3), (["delta", "gamma"], 3), (None, 9)])
def test_nombre_de_prix_calcules_par_greek(monkeypatch, greeks, nb_prix):
    """Le cout d'un bump : 2 prix par Greek, 3 pour le gamma, 9 pour les cinq."""
    compteur = {"n": 0}
    vrai_mc_price = estimators.mc_price

    def mc_price_compte(*args, **kwargs):
        compteur["n"] += 1
        return vrai_mc_price(*args, **kwargs)

    monkeypatch.setattr(estimators, "mc_price", mc_price_compte)
    bump_independant(S, K, r, T, sigma, "call", 1000, np.random.default_rng(0), greeks=greeks)
    assert compteur["n"] == nb_prix


# ------------------------------------------------------ regle de choix de h
def test_choisir_pas_plateau_prend_le_plus_petit_h():
    grille = [0.001, 0.01, 0.1, 1.0]
    valeurs = [1.2, 1.1, 1.0, 3.0]          # plateau ; minimum exact en h = 0.1
    h, v, h_min = choisir_pas(grille, valeurs)
    assert h == 0.001 and v == 1.2 and h_min == 0.1


def test_choisir_pas_creux_net_prend_le_creux():
    grille = [0.001, 0.01, 0.1, 1.0]
    valeurs = [50.0, 1.0, 1.5, 9.0]         # vrai creux en h = 0.01
    h, v, h_min = choisir_pas(grille, valeurs)
    assert h == 0.01 and h_min == 0.01


def test_choisir_pas_grille_non_triee_et_tolerance():
    grille = [1.0, 0.1, 0.01]
    valeurs = [1.0, 1.3, 1.1]
    assert choisir_pas(grille, valeurs)[0] == 0.01           # 1,1 <= 1,25
    assert choisir_pas(grille, valeurs, tolerance=0.05)[0] == 1.0
