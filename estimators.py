import numpy as np

from monte_carlo import mc_price, payoff, simulate_ST
   
# Calculate option price and Greeks using bump-and-revalue method.
#  Parameters:
   # S : float Current stock price.
    #K : float Strike price.
   # r : float Risk-free interest rate.
    #T : float Time to maturity (in years).
   # sigma : float Volatility of the underlying asset.
    #M : int Number of Monte Carlo simulations.
    # #rng : np.random.Generator Random number generator for reproducibility.
    # #commun : bool, optional If True, use common random numbers for bumping. Default is True.
   # h_S : float, optional Bump size for stock price. Default is 1.0.
   # h_sigma : float, optional Bump size for volatility. Default is 0.01.
   # h_r : float, optional Bump size for risk-free rate. Default is 0.001.
   # h_T : float, optional Bump size for time to maturity. Default is 0.01.
   # Returns:
   # dict: A dictionary containing the option price and Greeks (Delta, Gamma, Vega, Theta, Rho).

def bump_greeks(S, K, r, T, sigma, option_type, M, rng, commun=True,
                h_S=1.0, h_sigma=0.01, h_r=0.001, h_T=0.01):
    """Delta, gamma, vega, theta et rho par differences finies centrees.

    commun=True  : un seul tableau Z, reutilise pour tous les prix (nombres
                   aleatoires communs).
    commun=False : un nouveau tableau Z pour chaque prix (tirages independants).
    rng : generateur numpy, cree une seule fois (np.random.default_rng(graine)).
    """
    if commun:
        Z_commun = rng.standard_normal(M)

        def tirage():
            return Z_commun
    else:
        def tirage():
            return rng.standard_normal(M)

    def prix(S_, r_, T_, sigma_):
        return mc_price(S_, K, r_, T_, sigma_, tirage(), option_type)[0]

    V0 = prix(S, r, T, sigma)
    V_S_haut = prix(S + h_S, r, T, sigma)
    V_S_bas = prix(S - h_S, r, T, sigma)
    V_sigma_haut = prix(S, r, T, sigma + h_sigma)
    V_sigma_bas = prix(S, r, T, sigma - h_sigma)
    V_r_haut = prix(S, r + h_r, T, sigma)
    V_r_bas = prix(S, r - h_r, T, sigma)
    V_T_haut = prix(S, r, T + h_T, sigma)
    V_T_bas = prix(S, r, T - h_T, sigma)

    return {
        "delta": (V_S_haut - V_S_bas) / (2 * h_S),
        "gamma": (V_S_haut - 2 * V0 + V_S_bas) / h_S**2,
        "vega": (V_sigma_haut - V_sigma_bas) / (2 * h_sigma),
        "rho": (V_r_haut - V_r_bas) / (2 * h_r),
        "theta": -(V_T_haut - V_T_bas) / (2 * h_T),
    }


def bump_independant(S, K, r, T, sigma, option_type, M, rng, **pas):
    """Bump avec tirages independants : un nouveau Z pour chaque prix."""
    return bump_greeks(S, K, r, T, sigma, option_type, M, rng, commun=False, **pas)


def bump_commun(S, K, r, T, sigma, option_type, M, rng, **pas):
    """Bump avec nombres aleatoires communs : les memes Z pour tous les prix."""
    return bump_greeks(S, K, r, T, sigma, option_type, M, rng, commun=True, **pas)


def pathwise_greeks(S, K, r, T, sigma, option_type, Z):
    """Delta, vega et rho par pathwise (une seule simulation, pas de h).

    Pas de gamma (payoff non deux fois derivable) ni de theta (retenu par bump).
    """
    ST = simulate_ST(S, r, T, sigma, Z)

    if option_type == 'call':
        signe = 1
        indicatrice = ST > K
    elif option_type == 'put':
        signe = -1
        indicatrice = ST < K
    else:
        raise ValueError("Invalid option type. Please specify 'call' or 'put'.")

    actualisation = np.exp(-r * T)
    delta = signe * actualisation * np.mean(indicatrice * ST / S)
    vega = signe * actualisation * np.mean(indicatrice * ST * (np.sqrt(T) * Z - sigma * T))
    rho = signe * actualisation * np.mean(indicatrice * K * T)

    return {"delta": delta, "vega": vega, "rho": rho}


def likelihood_ratio_greeks(S, K, r, T, sigma, option_type, Z):
   

    ST = simulate_ST(S, r, T, sigma, Z)
    gain = payoff(ST, K, option_type)
    actualisation = np.exp(-r * T)
    racine_T = np.sqrt(T)

    poids_delta = Z / (S * sigma * racine_T)
    poids_gamma = (Z**2 - 1 - sigma * racine_T * Z) / (S**2 * sigma**2 * T)
    poids_vega = (Z**2 - 1) / sigma - racine_T * Z
    poids_rho = racine_T * Z / sigma

    prix = actualisation * np.mean(gain)

    return {
        "delta": actualisation * np.mean(gain * poids_delta),
        "gamma": actualisation * np.mean(gain * poids_gamma),
        "vega": actualisation * np.mean(gain * poids_vega),
        "rho": actualisation * np.mean(gain * poids_rho) - T * prix,
    }