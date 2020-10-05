import numpy as np
import pandas as pd
import scipy.stats as sp


class BlackScholes:
    """
    Black-Scholes model for pricing European options.
    
    Notes
    -----
    Assumptions of the model:

    - the volatility of the underlying asset is constant over time;
    - the underlying asset price follows the log-normal distribution;
    - the underlying assset can be traded continuously;
    - no transaction costs or taxes;
    - all securities are perfectly divisible;
    - the risk-free rate is constant and the same for all maturities.

    Price of the call option was calculated based on below formula:

    .. math:: c = S_{0}N(d_{1}) - Ke^{-rT}N(d_{2}) 

    On the other hand, price of the put option was derived from the formula:

    .. math:: p = Ke^{-rT}N(-d_{2}) - S_{0}N(-d_{1}),

    where:

    .. math:: d_{1} = \\frac{ ln(S_{0}/K) + (r + \\sigma^{2}/2)T }{ \\sigma \\sqrt{T} } \\
        d_{2} = \\frac{ ln(S_{0}/K) + (r - \\sigma^{2}/2)T }{ \\sigma \\sqrt{T} } = d_{1} - \\sigma \\sqrt{T}

    """

    def __init__(self, S: float, K: float, r: float, q: float=0, T: float, sigma: float, type: str) -> None:
        """
        Class initializer. 

        Parameters
        ----------
        S : float
            Price of the underlying.
        K : float
            Strike price.
        r : float
            Risk free rate.
        q : float
            Dividend rate.
        T : float
            Time to expiry (in years).
        sigma : float
            Underlying volatility.
        type : str
            Option type: "put" or "call".
        """    
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma
        self.type = type


    def d1(self) -> float:
        """
        Calculates d1 from the class description.

        Returns
        -------
        float
            d1 value
        """    
        return (np.log(self.S/self.K) + (self.r + (self.sigma**2)/2)*self.T) / (self.sigma*np.sqrt(self.T)) 


    def d2(self) -> float:
        """
        Calculates d2 from the class description.

        Returns
        -------
        float
            d2 value
        """        
        return self.d1() - self.sigma*np.sqrt(self.T)


    def price(self) -> float:
        """
        Runs Black-Scholes calculation.

        Returns
        -------
        float
            Option price
        """        
        if self.type == 'call':
            return self.S*sp.norm.cdf(self.d1()) - self.K*np.exp(-self.r*self.T)*sp.norm.cdf(self.d2())
        
        elif self.type == 'put':
            return self.K*np.exp(-self.r*self.T)*sp.norm.cdf(-self.d2()) - self.S*sp.norm.cdf(-self.d1())

        else:
            raise ValueError("Option price type can only be call or put")