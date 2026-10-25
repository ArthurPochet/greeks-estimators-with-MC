import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

import black_scholes as bs
from estimators import (bump_commun, bump_independant, bump_greeks, likelihood_ratio_greeks, pathwise_greeks)

S, K, r, T, sigma = 100, 100, 0.05, 1, 0.2 
M = 200_000

def greeks_exact(option_type):
    if option_type == 'call':
        return { "delta": bs.call_delta(S, K, r, T, sigma),
                 "gamma": bs.gamma(S, K, r, T, sigma),
                 "vega": bs.vega(S, K, r, T, sigma),
                 "theta": bs.call_theta(S, K, r, T, sigma),
                 "rho": bs.call_rho(S, K, r, T,sigma)}

    return { "delta": bs.put_delta(S, K, r, T, sigma),
              "gamma": bs.gamma(S, K, r, T, sigma),
              "vega": bs.vega(S, K, r, T, sigma),
              "theta": bs.put_theta(S, K, r, T, sigma),
              "rho": bs.put_rho(S, K, r, T, sigma)}

#tolerance abs (ordre grandeur, a ajuster en regardant les results)
tolerances = { "delta": 0.01,"gamma": 0.002, "vega": 1.0, "theta": 0.1, "rho": 1.0}

#Likelihood est plus bruité donc tolerance + larges
tolerances_LR = {"delta": 0.02, "gamma": 0.003, "vega": 3.0, "rho": 2.0}

                
#bump

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bump_commun_proche_de_BS(option_type):
    rng = np.random.default_rng(42)
    estime = bump_commun(S, K, r, T, sigma, option_type, M, rng)
    exact = greeks_exact(option_type)
    for nom in ["delta", "gamma", "vega", "theta", "rho"]:
        assert estime[nom] == pytest.approx(exact[nom], abs=tolerances[nom]), nom

def test_bump_reproductible():
    a = bump_commun(S, K, r, T, sigma, "call", 10_000, np.random.default_rng(7))
    b = bump_commun(S, K, r, T, sigma, "call", 10_000, np.random.default_rng(7))
    assert a == b

def test_bump_independant_renvoie_cinq_greeks():
    resultat = bump_independant(S, K, r, T, sigma, "call", 10_000, np.random.default_rng(1))
    assert set(resultat.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

#pathwise

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_pathwise_proche_BS(option_type):
    Z = np.random.default_rng(42).standard_normal(M)
    estime = pathwise_greeks(S, K, r, T, sigma, option_type, Z)
    exact = greeks_exact (option_type)
    for nom in ["delta", "vega", "rho"]:
        assert estime[nom] == pytest.approx(exact[nom], abs=tolerances[nom]), nom

def test_pathwise_pas_gamma_theta():
    Z = np.random.default_rng(42).standard_normal(1000)
    estime = pathwise_greeks(S, K, r, T, sigma, "call", Z)
    assert set(estime.keys()) == {"delta", "vega", "rho"}

def test_signes_pathwise_cp ():
    Z = np.random.default_rng(42).standard_normal(M)
    call = pathwise_greeks(S, K, r, T, sigma, "call", Z)
    put = pathwise_greeks ( S, K, r, T, sigma, "put", Z)
    assert 0 < call["delta"] < 1
    assert -1 < put["delta"] < 0
    assert call["rho"] > 0
    assert put["rho"] < 0

def test_option_type_invalide():
    with pytest.raises(ValueError):
        pathwise_greeks(S, K, r, T, sigma, "straddle", np.zeros(10))

#likelihood ration

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_likelihood_proche_BS(option_type):
    Z = np.random.default_rng(42).standard_normal(M)
    estime = likelihood_ratio_greeks(S, K, r, T, sigma, option_type, Z)
    exact = greeks_exact (option_type)
    for nom in ["delta", "gamma", "vega", "rho"]:
        assert estime[nom] == pytest.approx(exact[nom], abs=tolerances_LR[nom]), nom

def test_likelihood_ratio_donne_gamma_pas_theta():
    Z = np.random.default_rng(42).standard_normal(1000)
    estime = likelihood_ratio_greeks(S, K, r, T, sigma, "call", Z)
    assert set(estime.keys()) == {"delta", "gamma", "vega", "rho"}

def test_likelihood_ratio_gamma_pos_pour_cp():
    Z = np.random.default_rng(42).standard_normal(M)
    assert likelihood_ratio_greeks(S, K, r, T, sigma, "call", Z)["gamma"] > 0
    assert likelihood_ratio_greeks(S, K, r, T, sigma, "put", Z)["gamma"] > 0

# comparaison

def erreur_rms_delta(methode, n_graines=40, M_petit = 20_000):
    exact = bs.call_delta(S, K, r, T, sigma)
    erreur = []
    for graine in range(n_graines):
        rng = np.random.default_rng(graine)
        if methode == "bump_commun":
            delta = bump_commun(S, K, r, T, sigma, "call", M_petit, rng)["delta"]
        elif methode == "bump_independant":
            delta = bump_independant(S, K, r, T, sigma, "call", M_petit, rng)["delta"]
        elif methode == "pathwise":
            Z = rng.standard_normal(M_petit)
            delta = pathwise_greeks(S, K, r, T, sigma, "call", Z)["delta"]
        else: 
            Z = rng.standard_normal(M_petit)
            delta = likelihood_ratio_greeks(S, K, r, T, sigma, "call", Z)["delta"]
        erreur.append(delta-exact)
    return np.sqrt(np.mean(np.square(erreur)))

#hypo H2 en petit: m egal, les Z communs sont bien plus precis

def test_h2_bump_commun_plus_precis_que_bump_independant():
    assert erreur_rms_delta("bump_commun") < erreur_rms_delta ("bump_independant")/5

#Likelihood plus bruité que pathwise ? 
def test_likelihood_ration_moins_precis_que_pathwise_pour_delta():
    assert erreur_rms_delta("likelihood_ratio") > erreur_rms_delta("pathwise")


    
