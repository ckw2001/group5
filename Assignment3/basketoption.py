from math import sqrt, log, exp
from scipy.stats import norm
import numpy as np
import pandas as pd
class BasketOptionPricer:
    def __init__(self, S1,S2, K, T, sigma1,sigma2, r, rho, option_type):
        assert option_type in ('call', 'put'), "Option type must be 'call' or 'put'"
        self.S1 = S1
        self.S2 = S2
        self.K = K
        self.T = T
        self.sigma1=sigma1
        self.sigma2=sigma2
        self.r = r
        self.rho = rho  # Correlation matrix or scalar
        self.option_type = option_type


    def geometric_basket(self):
        n = 2  # Number of assets in basket
        B = (self.S1 * self.S2) ** (1 / n)  # Geometric Mean of Assets

        # Creating the covariance matrix
        sigma = [self.sigma1, self.sigma2]

        # Calculating basket volatility using the covariance matrix
        sigma_b = sqrt(self.sigma1**2 + self.sigma2**2 + 2 * self.rho * self.sigma1 * self.sigma2)/n

        # Adjusted drift term
        mu_b = self.r - 0.5 * sum(sigma[i] ** 2 for i in range(n)) / n + 0.5 * sigma_b ** 2

        # Black-Scholes d1 and d2 parameters
        d1 = (log(B / self.K) + (mu_b + 0.5 * sigma_b ** 2) * self.T) / (sigma_b * sqrt(self.T))
        d2 = d1 - sigma_b * sqrt(self.T)

        # Calculating option price based on option type
        if self.option_type == 'call':
            price = exp(-self.r * self.T) * (B * exp(mu_b * self.T) * norm.cdf(d1) - self.K * norm.cdf(d2))
        else:
            price = exp(-self.r * self.T) * (self.K * norm.cdf(-d2) - B * exp(mu_b * self.T) * norm.cdf(-d1))

        return price

    def price_path(self,M):
        np.random.seed(100)#Fixing the seed for reproducibility
        
        #Drift parameter of Brownian motion
        drift1 = (self.r- 0.5 *self.sigma1**2)*self.T 
        drift2 = (self.r- 0.5 *self.sigma2**2)*self.T 

        #Volatility parameter of Brownian motion
        vol1 = self.sigma1 * np.sqrt(self.T) 
        vol2 = self.sigma2 * np.sqrt(self.T) 

        #generate standard normal random variables with correlarion rho
        Z1 = np.random.randn(M,1)
        Z2 = Z1*self.rho+np.sqrt(1-self.rho**2)*np.random.randn(M,1)


        #generate price path
        S1Path = self.S1* np.exp(drift1 + (vol1*Z1)) 
        S2Path = self.S2* np.exp(drift2 + (vol2*Z2)) 
        return S1Path,S2Path
    

    def arithmetic_basket(self,M,control_variate=False):
        """
        Prices an arithmetic basket option using Monte Carlo simulation,
        with optional control variate technique.
        """
        S1Path, S2Path = self.price_path(M)
        
        # Calculate arithmetic basket option payoff
        if self.option_type == 'call':
            arithPayoff = np.exp(-self.r * self.T) * np.maximum((S1Path + S2Path) / 2  - self.K, 0)
            geoPayoff=np.exp(-self.r * self.T)* np.maximum(np.sqrt(S1Path * S2Path)- self.K,0)
        else:
            arithPayoff = np.exp(-self.r * self.T) * np.maximum(self.K -(S1Path + S2Path) / 2 , 0)
            geoPayoff=np.exp(-self.r * self.T)* np.maximum(self.K-np.sqrt(S1Path * S2Path),0)

        # Standard Monte Carlo without control variate
        if not control_variate:
            Pmean = np.mean(arithPayoff)
            Pstd = np.std(arithPayoff)
            confmc = [Pmean - 1.96 * Pstd / np.sqrt(M), Pmean + 1.96 * Pstd / np.sqrt(M)]
            return Pmean, confmc

        # Control variate using geometric basket price
        else:
            geoPrice = self.geometric_basket()  # Closed form solution for geometric basket
            # Covariance and variance for the control variate technique
            covXY = np.mean(np.multiply(arithPayoff,geoPayoff))- np.mean(arithPayoff)*np.mean(geoPayoff) 
            theta = covXY / np.var(geoPayoff)
            Z = arithPayoff + theta * (geoPrice - geoPayoff)
            Zmean = np.mean(Z)
            Zstd = np.std(Z)
            confcv = [Zmean - 1.96 * Zstd / np.sqrt(M), Zmean + 1.96 * Zstd / np.sqrt(M)]
            return Zmean, confcv




def test_basket_option_pricer():
    parameters = [
        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.5, 'type': 'put'},
        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.9, 'type': 'put'},
        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.1, 'sigma2': 0.3, 'rho': 0.5, 'type': 'put'},
        {'S1': 100, 'S2': 100, 'K': 80,  'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.5, 'type': 'put'},
        {'S1': 100, 'S2': 100, 'K': 120, 'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.5, 'type': 'put'},
        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.5, 'sigma2': 0.5, 'rho': 0.5, 'type': 'put'},

        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.5, 'type': 'call'},
        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.9, 'type': 'call'},
        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.1, 'sigma2': 0.3, 'rho': 0.5, 'type': 'call'},
        {'S1': 100, 'S2': 100, 'K': 80,  'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.5, 'type': 'call'},
        {'S1': 100, 'S2': 100, 'K': 120, 'sigma1': 0.3, 'sigma2': 0.3, 'rho': 0.5, 'type': 'call'},
        {'S1': 100, 'S2': 100, 'K': 100, 'sigma1': 0.5, 'sigma2': 0.5, 'rho': 0.5, 'type': 'call'}
    ]

    # Results list
    results = []
    # Test each set of parameters
    for params in parameters:
        pricer = BasketOptionPricer(
            S1=params['S1'],
            S2=params['S2'],
            K=params['K'],
            T=3,
            sigma1=params['sigma1'],
            sigma2=params['sigma2'],  
            r=0.05,
            rho=params['rho'],
            option_type=params['type'],
        )
        M = 100000

        # Compute arithmetic basket option price without control variant
        price_without_cv, conf_without_cv = pricer.arithmetic_basket(M, control_variate=False)
        
        # Compute arithmetic basket option price with control variant
        price_with_cv, conf_with_cv = pricer.arithmetic_basket(M, control_variate=True)
        
        # Compute geometric basket option price
        geometric_price = pricer.geometric_basket()
        
        # Add results to the list
        results.append({
            'S1': params['S1'],
            'S2': params['S2'],
            'K':  params['K'],
            'T':  3,
            'sigma1': params['sigma1'],
            'sigma2': params['sigma2'],
            'r': 0.05,
            'rho': params['rho'],
            'option_type': params['type'],
            'M': 100000,
            'Geometric': geometric_price,
            'Arithmetic MC': price_without_cv,
            'MC Confidence Interval': conf_without_cv,
            'Arithmetic Control Variate': price_with_cv,
            'Control Variate Confidence Interval': conf_with_cv
        })
    
    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    
    # Display the DataFrame
    display(results_df)
    
    #Output the DataFrame to a CSV file
    results_df.to_csv('option_pricing_results.csv', index=False)
    
    return results_df

# 使用Pandas DataFrame显示测试结果
test_basket_option_pricer_df = test_basket_option_pricer()
