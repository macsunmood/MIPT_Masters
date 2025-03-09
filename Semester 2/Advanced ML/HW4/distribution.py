import numpy as np


class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        med = np.median(x, axis=0)
        mad = np.mean(np.abs(x - med), axis=0)  # mean absolute deviation from the median for each feature
        return mad
        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        # Calculate location and scale parameters for Laplace distribution
        self.loc = np.median(features, axis=0)  # location (median for each feature) -- MLE estimator of 𝜇
        self.scale = self.mean_abs_deviation_from_median(features)  # scale (mean absolute deviation from the median) -- MLE estimator of 𝑏
        ####

    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block

        # Taking natural logarithm of Probability Density Function: 
        # log(1 / 2𝑏 * exp(-|x - μ| / 𝑏) = -log(2𝑏) - |x - μ| / 𝑏, 
        # where 𝑏 is the scale and μ is the location parameter
        abs_diff = np.abs(values - self.loc)  # absolute difference between the values and the location
        log_norm_const = -np.log(2 * self.scale)  # logarithm of the normalization constant: log(1 / (2 * scale))
        return log_norm_const - abs_diff / self.scale
        ####
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))  # Probability Density Function of Laplace distribution
