import math
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
import json


def quasi_mc_kiko(s,sigma,r,T,barrier_lower,barrier_upper,N,R):
    '''
    This function uses the quasi-Monte Carlo method to simulate the price and the Delta of giving KIKO put option
    :param s:  spot price of asset
    :param sigma: volatility
    :param r: risk-free interest rate
    :param T: time to maturity (in years)
    :param barrier_lower: knock in lower bound
    :param barrier_upper: knock out upper bound
    :param N:the number of steps
    :param R:rebate payment
    :return:    {
    "price": "5.9880",
    "Delta": "2.0427"
    }
    '''


    # delta t
    deltaT = T / N
    # set the random seed
    seed = 1000
    np.random.seed(seed)
    # generate the paths of stock prices
    values = []
    # the number of simulation
    M = int(1e4)
    sequencer = qmc.Sobol(d=N, seed=seed)
    # uniform samples
    X = np.array(sequencer.random(n=M))
    # standard normal samples
    Z = norm.ppf(X)
    # scaled samples
    samples = (r - 0.5 * sigma * sigma) * deltaT + sigma * math.sqrt(deltaT) * Z
    df_samples = pd.DataFrame(samples)
    df_samples_cumsum = df_samples.cumsum(axis=1)
    # the simulated stock prices, M rows, N columns
    df_stocks = s * np.exp(df_samples_cumsum)
    for ipath in df_stocks.index.to_list():
        ds_path_local = df_stocks.loc[ipath, :]
        price_max = ds_path_local.max()
        price_min = ds_path_local.min()
        if price_max >= barrier_upper:  # knock-out happened
            knockout_time = ds_path_local[ds_path_local
                                          >= barrier_upper].index.to_list()[0]
            payoff = R * np.exp(-knockout_time * r * deltaT)
            values.append(payoff)
        elif price_min <= barrier_lower:  # knock-in happend
            final_price = ds_path_local.iloc[-1]
            payoff = np.exp(- r * T) * max(K - final_price, 0)
            values.append(payoff)
        else:  # no knock-out, no knock-in
            values.append(0)
    # print(values)
    option_delta = (max(values) - min(values)) / (price_max - price_min)
    value = np.mean(values)

    result ={
        'price':f'{value:.4f}',
        'Delta':f'{option_delta:.4f}'
    }

    # std = np.std(values)
    # conf_interval_lower = value - 1.96 * std / math.sqrt(M)
    # conf_interval_upper = value + 1.96 * std / math.sqrt(M)
    """
    return example:
    {
    "price": "5.9880",
    "Delta": "2.0427"
    }

    """
    return json.dumps(result, indent=4)





# ================================
test
r = 0.05
sigma = 0.20
T = 2.0
s = 100
K = 100
barrier_lower = 80
barrier_upper = 125
N = 24
R = 1.5
print(quasi_mc_kiko(s,sigma,r,T,barrier_lower,barrier_upper,N,R))


