import numpy as np
from scipy.stats import norm
import pandas as pd




class AsianOptionPricer:
    """
    A class for pricing Asian options using Monte Carlo simulation, with an
    option to use the control variate technique based on the geometric Asian option.
    """
    def __init__(self, S0, sigma, r, T, K, n, option_type):
        """
        Initializes the pricer with given parameters.
        """
        self.S0 = S0  # Initial stock price
        self.sigma = sigma  # Volatility of the stock
        self.r = r  # Risk-free interest rate
        self.T = T  # Time to maturity of the option
        self.K = K  # Strike price of the option
        self.n = n  # Number of observation times
        self.option_type = option_type  # 'call' or 'put'

    def generate_price_paths(self, M, dt):
        """
        Generates simulated price paths for the underlying asset using Geometric Brownian Motion.
        """
        np.random.seed(100)  # Fixing the seed for reproducibility
        drift = np.exp((self.r - 0.5 * self.sigma**2) * dt)
        Z = np.random.randn(M, self.n)
        growth_factors = drift * np.exp(self.sigma * np.sqrt(dt) * Z)
        S_paths = self.S0 * np.cumprod(growth_factors, axis=1)
        return S_paths






    def price_geometric_option(self):
        """
        Prices a geometric Asian option using the closed-form solution.
        """
        dt = self.T / self.n
        sigsqT = self.sigma**2 * self.T * (self.n + 1) * (2 * self.n + 1) / (6 * self.n* self.n)
        muT = (self.r - 0.5 * self.sigma**2) * self.T * (self.n + 1) / (2*self.n)+0.5*sigsqT
        d1 = (np.log(self.S0 / self.K) + (muT + 0.5 * sigsqT)) / np.sqrt(sigsqT)
        d2 = d1 - np.sqrt(sigsqT)
        if self.option_type == 'call':
            N1 = norm.cdf(d1)
            N2 = norm.cdf(d2)
            price = np.exp(-self.r * self.T) * (self.S0 * np.exp(muT) * N1 - self.K * N2)
        else:
            N1 = norm.cdf(-d1)
            N2 = norm.cdf(-d2)
            price = np.exp(-self.r * self.T) * (self.K * N2 - self.S0 * np.exp(muT) * N1)
        return {'geometric_price': price}





    def price_arithmetic_option(self, M, use_control_variate=True):
        """
        Prices an arithmetic Asian option using Monte Carlo simulation.
        """
        dt = self.T / self.n
        paths = self.generate_price_paths(M, dt)
        arithmetic_means = np.mean(paths, axis=1)
        geometric_means = np.exp(np.sum(np.log(paths), axis=1) / self.n)

        if self.option_type == 'call':
            arithPayoff = np.exp(-self.r * self.T) * np.maximum(arithmetic_means - self.K, 0)
            geometric_payoffs=np.exp(-self.r * self.T) * np.maximum(geometric_means - self.K, 0)
        else:
            arithPayoff = np.exp(-self.r * self.T) * np.maximum(self.K - arithmetic_means, 0)
            geometric_payoffs = np.exp(-self.r * self.T) * np.maximum(self.K - geometric_means, 0)
        

        if not use_control_variate:
            Pmean = np.mean(arithPayoff)
            Pstd = np.std(arithPayoff)
            confmc = [Pmean - 1.96 * Pstd / np.sqrt(M), Pmean + 1.96 * Pstd / np.sqrt(M)]
            return {'arithmetic_price': Pmean, 'confidence_interval': confmc}





        else :
            geometric_price = self.price_geometric_option()['geometric_price']
            cov_xy =  np.mean(np.multiply(arithPayoff,geometric_payoffs))- np.mean(arithPayoff)*np.mean(geometric_payoffs) 
            theta = cov_xy / np.var(geometric_payoffs)
            Z=arithPayoff + theta * (geometric_price - geometric_payoffs)

            Zmean = np.mean(Z)
            Zstd = np.std(Z)
            confcv = [Zmean - 1.96 * Zstd / np.sqrt(M), Zmean + 1.96 * Zstd / np.sqrt(M)]
            return {'arithmetic_price': Zmean, 'confidence_interval': confcv}

# Function to run test cases
def test_option_pricing():
    test_cases = [
        # Test cases are dictionaries with parameters for each option to test
        {'sigma': 0.3, 'K': 100, 'n': 50, 'option_type': 'put', 'r': 0.05, 'T': 3, 'S0': 100},
        {'sigma': 0.3, 'K': 100, 'n': 100, 'option_type': 'put', 'r': 0.05, 'T': 3, 'S0': 100},
        {'sigma': 0.4, 'K': 100, 'n': 50, 'option_type': 'put', 'r': 0.05, 'T': 3, 'S0': 100},
        {'sigma': 0.3, 'K': 100, 'n': 50, 'option_type': 'call', 'r': 0.05, 'T': 3, 'S0': 100},
        {'sigma': 0.3, 'K': 100, 'n': 100, 'option_type': 'call', 'r': 0.05, 'T': 3, 'S0': 100},
        {'sigma': 0.4, 'K': 100, 'n': 50, 'option_type': 'call', 'r': 0.05, 'T': 3, 'S0': 100}
    ]
    M = 100000  # Number of simulation paths

    results = {'Sigma': [], 'K': [], 'n': [], 'Option Type': [], 'Geometric': [],
               'Arithmetic Without CV': [], 'Interval Without CV': [], 'Arithmetic_cv': [],
               'Confidence Interval_cv': []}

    # Iterate through each test case and price the options
    for case in test_cases:
        pricer = AsianOptionPricer(**case)
        geometric_price = pricer.price_geometric_option()['geometric_price']
        
        arithmetic_result_without_cv = pricer.price_arithmetic_option(M, use_control_variate=False)
        arithmetic_price_without_cv = arithmetic_result_without_cv['arithmetic_price']
        arithmetic_interval_without_cv = arithmetic_result_without_cv['confidence_interval']

        arithmetic_result_with_cv = pricer.price_arithmetic_option(M, use_control_variate=True)
        arithmetic_price_with_cv = arithmetic_result_with_cv['arithmetic_price']
        arithmetic_interval_with_cv = arithmetic_result_with_cv['confidence_interval']

        # Store the results
        results['Sigma'].append(case['sigma'])
        results['K'].append(case['K'])
        results['n'].append(case['n'])
        results['Option Type'].append(case['option_type'])
        results['Geometric'].append(geometric_price)
        results['Arithmetic Without CV'].append(arithmetic_price_without_cv)
        results['Interval Without CV'].append(arithmetic_interval_without_cv)
        results['Arithmetic_cv'].append(arithmetic_price_with_cv)
        results['Confidence Interval_cv'].append(arithmetic_interval_with_cv)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    df.to_csv('asainoption.csv', index=False)    

    return df

# Run the test cases
print(test_option_pricing())






