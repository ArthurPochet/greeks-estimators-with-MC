#Monte Carlo EU option
#Z = tirages normaux standard

import math 
import numpy as np

def simulate_ST(S, r, T, sigma, Z):
    ST = S*np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    return ST

def payoff(ST, K, option_type):
    if option_type == 'call':
        return np.maximum(ST-K,0)
    elif option_type == 'put':
        return np.maximum (K-ST,0)
    else:
        raise ValueError("Invalid option type. Please specify 'call' or 'put'.")

#raise est une exception dans le cas ou quelque chose ne respect aps les deux conditions

def mc_price(S, K, r, T, sigma, Z, option_type):
    ST = simulate_ST(S, r, T, sigma, Z)
    gain_actualisé = np.exp(-r*T)*payoff(ST, K, option_type)
    M = len(Z)
    price = gain_actualisé.mean()
    st_dev =  gain_actualisé.std(ddof=1)/np.sqrt(M)
    return price, st_dev   