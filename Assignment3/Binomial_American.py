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
    dt = T / N

    # Calculate the up and down factors
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Initialize the tree and option values arrays
    tree = np.zeros((N + 1, N + 1))
    option_values = np.zeros((N + 1, N + 1))

    # Set the initial asset price at the root of the tree
    tree[0, 0] = S0

    # Build the binomial tree
    for t in range(1, N + 1):
        for x in range(t + 1):
            if x == 0:
                tree[t, x] = tree[t - 1, x] * d
            else:
                tree[t, x] = tree[t - 1, x - 1] * u

    # Calculate option values at maturity
    for x in range(N + 1):
        if option_type == 'call':
            option_values[N, x] = max(tree[N, x] - K, 0)
        elif option_type == 'put':
            option_values[N, x] = max(K - tree[N, x], 0)

    # Use backward induction to calculate option values at each node
    for t in range(N - 1, -1, -1):
        for x in range(t + 1):
            # Calculate the continuation value
            continuation_value = (p * option_values[t + 1, x] + (1 - p) * option_values[t + 1, x + 1]) * math.exp(
                -r * dt)

            # Calculate the early exercise value
            if option_type == 'call':
                early_exercise_value = max(tree[t, x] - K, 0)
            elif option_type == 'put':
                early_exercise_value = max(K - tree[t, x], 0)

            # Choose the greater value: early exercise or continuation
            option_values[t, x] = max(early_exercise_value, continuation_value)

    # The price of the option is at the root of the tree

    result = {
        'price':f'{option_values[0][0]:.4f}'
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