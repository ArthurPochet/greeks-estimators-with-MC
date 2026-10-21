import math
from scipy.stats import norm

def d1(S, K, r, T, sigma):
    return (math.log(S/K) + (r + sigma**2/2)*T)/ (sigma*math.sqrt(T))

def d2(S, K, r, T, sigma):
    d2 = d1(S, K, r, T, sigma) - sigma*math.sqrt(T)
    return d2

#prix
def call_price(S, K, r, T, sigma):
    return (S*norm.cdf(d1(S, K, r, T, sigma)) - K*math.exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma)))

def put_price(S, K, r, T, sigma):
    return (K*math.exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma)) - S*norm.cdf(-d1(S, K, r, T, sigma)))

#Delta

def call_delta(S, K, r, T, sigma):
    return norm.cdf(d1(S, K, r, T, sigma))

def put_delta(S, K, r, T, sigma):
    return norm.cdf(d1(S, K, r, T, sigma)) - 1

# Gamma and Vega

def gamma(S, K, r, T, sigma):
    return norm.pdf(d1(S, K, r, T, sigma)) / (S*sigma*math.sqrt(T))

def vega(S, K, r, T, sigma):
    return S*norm.pdf(d1(S, K, r, T, sigma))*math.sqrt(T)

# Theta (one year)

def call_theta(S, K, r, T, sigma):
    term_vol = (-S*norm.pdf(d1(S, K, r, T, sigma))*sigma/(2*math.sqrt(T)))
    term_int = (-r*K*math.exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma)))
    return term_vol + term_int

def put_theta(S, K, r, T, sigma):
    term_vol = (-S*norm.pdf(d1(S, K, r, T, sigma))*sigma/(2*math.sqrt(T)))
    term_int = (r*K*math.exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma)))
    return term_vol + term_int 

# Rho (per unit of interest rate)

def call_rho(S, K, r, T, sigma):
    return K*T*math.exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma))

def put_rho(S, K, r, T, sigma):
    return -K*T*math.exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma))
    
    