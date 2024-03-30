import numpy as np
from scipy.stats import norm

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

    def _generate_price_paths(self, M, dt):
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
        return price





    def price_arithmetic_option(self, M, use_control_variate=True):
        """
        Prices an arithmetic Asian option using Monte Carlo simulation.
        """
        dt = self.T / self.n
        paths = self._generate_price_paths(M, dt)
        arithmetic_means = np.mean(paths, axis=1)
        geometric_means = np.exp(np.sum(np.log(paths), axis=1) / self.n)

        if self.option_type == 'call':
            payoffs = np.exp(-self.r * self.T) * np.maximum(arithmetic_means - self.K, 0)
            geometric_payoffs=np.exp(-self.r * self.T) * np.maximum(geometric_means - self.K, 0)
        else:
            payoffs = np.exp(-self.r * self.T) * np.maximum(self.K - arithmetic_means, 0)
            geometric_payoffs = np.exp(-self.r * self.T) * np.maximum(self.K - geometric_means, 0)
        
        if use_control_variate:
            geometric_price = self.price_geometric_option()
            cov_xy =  np.mean(np.multiply(payoffs,geometric_payoffs))- np.mean(payoffs)*np.mean(geometric_payoffs) 
            theta = cov_xy / np.var(geometric_payoffs)
            payoffs -= theta * (geometric_payoffs - geometric_price)

        price_estimate = np.mean(payoffs)
        price_std = np.std(payoffs)
        confidence_interval = [
            price_estimate - 1.96 * price_std / np.sqrt(M),
            price_estimate + 1.96 * price_std / np.sqrt(M)
        ]
        return price_estimate, confidence_interval

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

    # Print the header for the output
    print(f"{'sigma':<10}{'K':<10}{'n':<10}{'option_type':<15}{'Geometric':<20}{'Arithmetic MC':<20}{'MC Confidence Interval':<40}{'Arithmetic Control Variate':<30}{'Control Variate Confidence Interval':<40}")

    # Iterate through each test case and price the options
    for case in test_cases:
        pricer = AsianOptionPricer(**case)
        geometric_price = pricer.price_geometric_option()
        arithmetic_price, arithmetic_confidence = pricer.price_arithmetic_option(M, False)
        arithmetic_price_cv, arithmetic_confidence_cv = pricer.price_arithmetic_option(M, True)
        
        # Format the output as specified
        print(f"{case['sigma']:<10}{case['K']:<10}{case['n']:<10}{case['option_type']:<15}{geometric_price:<20.4f}{arithmetic_price:<20.4f}{'[' + ', '.join(f'{i:.4f}' for i in arithmetic_confidence) + ']':<40}{arithmetic_price_cv:<30.4f}{'[' + ', '.join(f'{i:.4f}' for i in arithmetic_confidence_cv) + ']':<40}")

# Run the test cases
test_option_pricing()
