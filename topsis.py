import numpy as np
import pandas as pd


class TOPSIS:
    """
    TOPSIS algorithm that is used to alternatives based on a given
    criterion set. The whole process in encapsulated in 5 steps
    listed as a comments in the body of the class. 

    Parameters
    ----------
    distance : {'euclidean', 'manhattan', 'chebyshev'}, default='euclidean'
        The distance function used in calculation.

    Attributes
    ----------
    score : ndarray
        TOPSIS score for each alternative.
    rank : ndarray
        Descending rank for each alternative based on score.
    """    

    def __init__(self, distance: str='euclidean') -> None:
        self.distance = distance
        
    def fit(self, eval_matrix: pd.DataFrame, weights: ndarray) -> None:
        """
        Runs TOPSIS algorithm.

        Parameters
        ----------
        eval_matrix : pd.DataFrame
            Evaluation matrix M x N, where M-alternatives and N-criterions
        weights : ndarray
            Vector with weight for each criterion
        """

        #1 Evaluation matrix normalization
        eval_matrix = eval_matrix / np.sqrt(np.sum(eval_matrix**2, axis=0))

        #2 Weighted decision matrix normalization
        weights = weights / np.sum(weights)
        eval_matrix = eval_matrix * weights

        #3 Best and worst alternative determination
        best = np.max(eval_matrix, axis=0)
        worst = np.min(eval_matrix, axis=0)

        #4 Distance calculation
        if self.distance == 'euclidean':
            d_b = np.sqrt(np.sum((eval_matrix - best)**2, axis=1))
            d_w = np.sqrt(np.sum((eval_matrix - worst)**2, axis=1))
        elif self.distance == 'manhattan':
            d_b = np.sum(abs(eval_matrix - best), axis=1)
            d_w = np.sum(abs(eval_matrix - worst), axis=1)
        elif self.distance == 'chebyshev':
            d_b = np.max(abs(eval_matrix - best), axis=1)
            d_w = np.max(abs(eval_matrix - worst), axis=1)
        else:
            raise ValueError("Possible values of 'distance' parameter are ['euclidean', 'manhattan', ''chebyshev]")

        #5 Score calculation and ranking
        self.score = d_w / (d_w + d_b)
        self.rank = np.argsort(self.score)[::-1] 