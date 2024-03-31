import json

def binomial_tree_american_option(S0, volatility, r, T, K, N, option_type):
    """
    Calculates the price of an American call or put option using the Binomial Tree method.

    Args:
        S0 (float): Spot price of the asset at time 0.
        volatility (float): Volatility of the asset.
        r (float): Risk-free interest rate.
        T (float): Time to maturity (in years).
        K (float): Strike price.
        N (int): Number of steps in the binomial tree.
        option_type (str): 'call' for call option, 'put' for put option.

    Returns:
        float: json
        {
        "price": "2.2964"
        }
    """
    dt = T / N  # Time step
    u = 1 + volatility * dt**0.5  # Up factor
    d = 1 - volatility * dt**0.5  # Down factor
    p = (1 + r * dt - d) / (u - d)  # Risk-neutral probability

    # Initialize option values at maturity
    option_values = [0] * (N + 1)
    for i in range(N + 1):
        if option_type == 'call':
            option_values[i] = max(0, S0 * u**i * d**(N - i) - K)
        elif option_type == 'put':
            option_values[i] = max(0, K - S0 * u**i * d**(N - i))

    # Backward induction
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            if option_type == 'call':
                option_values[i] = max(
                    S0 * u**i * d**(j - i) - K,
                    (p * option_values[i] + (1 - p) * option_values[i + 1]) / (1 + r * dt)
                )
            elif option_type == 'put':
                option_values[i] = max(
                    K - S0 * u**i * d**(j - i),
                    (p * option_values[i] + (1 - p) * option_values[i + 1]) / (1 + r * dt)
                )
    result = {
        'price':f'{option_values[0]:.4f}'
    }
    # return format:
    # {
    #     "price": "2.2964"
    # }

    return json.dumps(result, indent=4)

# example
S0 = 50
volatility = 0.2
r = 0.05
T = 0.5
K = 50
N = 200
call_option_price = binomial_tree_american_option(S0, volatility, r, T, K, N, option_type='call')
put_option_price = binomial_tree_american_option(S0, volatility, r, T, K, N, option_type='put')

print(call_option_price)