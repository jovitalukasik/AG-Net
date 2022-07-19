######################################################################################
# Based on
# Copyright (c) Colin White, Naszilla, 
# https://github.com/naszilla/naszilla
# modified
######################################################################################

import torch 
import numpy as np
from torch.distributions import Normal
from scipy.stats import norm


##############################################################################
#
#                              Arguments
#
##############################################################################


##############################################################################
#
#                   Acquisition Functions for BO
#
##############################################################################

def acquisition_fct(mu, sigma, best, type):
    if type == 'ei':
        value = expected_improvement(mu, sigma, best)
    
    elif type == 'pi':
        value = probability_improvement(mu, sigma, best)
    
    elif type == 'ucb':
        value = upper_confidence_bound(mu, sigma)
    
    elif type == 'ts':
        value = thompson_sampling(mu, sigma, best)
    elif type == 'its':
        value = independent_thompson_sampling(mu, sigma)
    else:
        raise TypeError("Unknow Acqusition Function type: {}".format(type))

    return value

##############################################################################
#
#                         Expected Improvement
#
##############################################################################
def expected_improvement(mu, v, best):
    u = (best-mu)/v
    ei = v * (u * norm.cdf(u) + norm.pdf(u))
    sorted_indices = np.argsort(ei)
    return sorted_indices

##############################################################################
#
#                         Probability Improvement
#
##############################################################################
def probability_improvement(mu, sigma, best):
    u = (mu-best)/sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    prob = normal.cdf(u)
    sorted_indices = torch.argsort(prob)

    return sorted_indices


##############################################################################
#
#                         Thompson Sampling
#
##############################################################################
def thompson_sampling(mu, sigma, best):
    print('check how this works, still debugging')
    return ts


##############################################################################
#
#                         Upper Confidence Bound
#
##############################################################################
def upper_confidence_bound(mu, sigma):
    exploration_factor = 0.5
    ucb = mu - exploration_factor*sigma
    sorted_indices = torch.argsort(ucb)
    return sorted_indices


##############################################################################
#
#           Independent Thompson sampling (ITS) acquisition function
#
##############################################################################
def independent_thompson_sampling(mean,sigma):
    samples = np.random.normal(mean, sigma)
    sorted_indices = np.argsort(samples)
    return sorted_indices