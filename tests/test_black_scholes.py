import math 
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from black_scholes import (
    call_price, put_price, call_delta, put_delta, gamma, vega,
    call_theta, put_theta, call_rho, put_rho,
)

#data for testing
S, K, r, T, sigma =  100, 100, 0.05, 1, 0.2 

def test_call_price():
    assert call_price(S, K, r, T, sigma) == pytest.approx(10.4506, abs=1e-3)

def test_put_price():
    assert put_price(S, K, r, T, sigma) == pytest.approx(5.5735, abs=1e-3)

def test_call_delta():
    assert call_delta(S, K, r, T, sigma) == pytest.approx(0.6368, abs=1e-3)

def test_put_delta():
    assert put_delta(S, K, r, T, sigma) == pytest.approx(-0.3632, abs=1e-3)

def test_gamma():
    assert gamma(S, K, r, T, sigma) == pytest.approx(0.018762, abs=1e-5)

def test_vega():
    assert vega(S, K, r, T, sigma) == pytest.approx(37.524, abs=1e-2)

def test_call_theta():
    assert call_theta(S, K, r, T, sigma) == pytest.approx(-6.414, abs=1e-2)

def test_put_theta():
    assert put_theta(S, K, r, T, sigma) == pytest.approx(-1.658, abs=1e-2)

def test_call_rho():
    assert call_rho(S, K, r, T, sigma) == pytest.approx(53.232, abs=1e-2)

def test_put_rho():
    assert put_rho(S, K, r, T, sigma) == pytest.approx(-41.89, abs=1e-2)

def test_call_put_parity():
    c = call_price(S, K, r, T, sigma)
    p = put_price(S, K, r, T, sigma)
    assert c - p == pytest.approx(S-K*math.exp(-r*T), abs=1e-9)

def test_delta_bornes():
    assert 0 <= call_delta(S, K, r, T, sigma) <= 1
    assert -1 <= put_delta(S, K, r, T, sigma) <= 0

def test_signs_gamma_vega_rho():
    assert gamma(S, K, r, T, sigma) > 0
    assert vega(S, K, r, T, sigma) > 0
    assert call_rho(S, K, r, T, sigma) > 0
    assert put_rho(S, K, r, T, sigma) < 0
