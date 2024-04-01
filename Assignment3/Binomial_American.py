import numpy as np
import math
import json


def binomial_tree_american_option(S0, sigma, r, T, K, N, option_type='call'):
    """
    Calculate the price of an American call/put option using the binomial tree method.

    Parameters:
    S0 (float): The spot price of the asset (at time 0).
    sigma (float): The volatility of the asset.
    r (float): The risk-free interest rate.
    T (float): The time to maturity (in years).
    K (float): The strike price of the option.
    N (int): The number of steps in the binomial tree.
    option_type (str): The type of option, 'call' or 'put'. Default is 'call'.

    Returns:
    float: The estimated price of the American option.
    """

    # Calculate the time step
    dt = T/N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1/u
    S = np.zeros((N+1,N+1))
    S[0,0] = S0
    p = (math.exp(r*dt) - d)/(u-d)
    # Use backward induction to calculate option values at each node
    for i in range(1,N+1):
        for a in range(i):
            S[a,i] = S[a,i-1] * u
            S[a+1,i] = S[a,i-1] * d

    Sv = np.zeros_like(S)
    if option_type == "call":
        S_intrinsic = np.maximum(S-K,0)
    else:
        S_intrinsic = np.maximum(K-S,0)
    Sv[:,-1] = S_intrinsic[:,-1]
    for i in range(N-1,-1,-1):
        for a in range(i+1):
            Sv[a,i] = max((Sv[a,i+1] * p + Sv[a+1,i+1] * (1-p))/np.exp(r*dt),S_intrinsic[a,i])


    # The price of the option is at the root of the tree

    result = {
        'price':f'{Sv[0,0]:.4f}'
    }
    # return format:
    # {
    #     "price": "2.2964"
    # }

    return json.dumps(result, indent=4)




# example
#S0 = 50
#volatility = 0.2
#r = 0.05
#T = 0.5
#K = 50
#N = 200
#call_option_price = binomial_tree_american_option(S0, volatility, r, T, K, N, option_type='call')
#put_option_price = binomial_tree_american_option(S0, volatility, r, T, K, N, option_type='put')

#print(call_option_price)