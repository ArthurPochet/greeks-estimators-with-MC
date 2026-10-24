import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from black_scholes import call_price, put_price
from monte_carlo import simulate_ST, payoff, mc_price

S, K, r, T, sigma = 100, 100, 0.05, 1, 0.2
M = 100_000

def tirer_Z(graine = 42, M=M):
    rng = np.random.default_rng(graine)
    return rng.standard_normal(M) 

def test_simulate_ST():
    Z = tirer_Z(M=1000)
    ST = simulate_ST(S, r, T, sigma, Z)
    assert ST.shape == Z.shape
    assert np.all(ST > 0)

def test_simulate_ST_Z_null():
    ST = simulate_ST(S, r, T, sigma, np.array([0.0]))
    assert ST[0] == pytest.approx(S*math.exp((r-sigma**2/2)*T))

def test_price_call_near_BS():
    price, se = mc_price(S, K, r, T, sigma, tirer_Z(), 'call')
    assert abs(price - call_price(S, K, r, T, sigma)) < 3*se

def test_price_put_near_BS():
    price, se = mc_price(S, K, r, T, sigma, tirer_Z(), 'put')
    assert abs(price - put_price(S, K, r, T, sigma)) < 3*se

def test_same_Z_same_results():
    Z = tirer_Z()
    assert mc_price(S, K, r, T, sigma, Z, "call") == mc_price(S,K, r, T, sigma, Z, "call")

def test_call_put_parity_same_Z():
    Z =tirer_Z()
    c, _ = mc_price(S, K, r, T, sigma, Z, "call")
    p, _ = mc_price(S, K, r, T, sigma, Z, "put")
    assert c - p == pytest.approx(S-K*math.exp(-r*T), abs= 0.1)

def test_st_dev_decrease_with_M():
    _, se_petit = mc_price(S, K, r, T, sigma, tirer_Z(M=1000), 'call')
    _, se_grand = mc_price(S, K, r, T, sigma, tirer_Z(M=100000), "call")
    assert se_grand < se_petit

def test_interval_confiance_95():
    exact = call_price(S, K, r, T, sigma)
    dedans = 0
    for graine in range(200):
        prix, se = mc_price (S, K, r, T, sigma, tirer_Z(graine, 10000), "call")
        if abs (prix - exact) < 2*se:
            dedans += 1
    assert 0.90 <= dedans/200 <= 0.99







