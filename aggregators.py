# -*- coding: utf-8 -*-
"""
Aggregator
"""

import sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
class additive_scalar_aggregator:
    compute f(c(e(x))) where e() is encoder, c() is concept, and f() is this class
    just aggregates a set of concept representations and their scores (parametrizer), and
    generates a prediction probability output from them.
def __init__:
    initialization 
def forward:
    compute f( c(e(x)), theta(e(x)) ) where e() is encoder, c() is concept, theta() is parametrizer and f() is this class 
NOTE: This function is not modified by Sawada
"""
class additive_scalar_aggregator(nn.Module):
    """ 
    def __init__:
        initialization 
    Inputs:
        cdim (int): input concept dimension
        nclasses (int): number of target classes
    Return:
        None
    """
    def __init__(self, cdim, nclasses):
        super(additive_scalar_aggregator, self).__init__()
        self.cdim      = cdim       # Dimension of each concept
        self.nclasses  = nclasses   # Numer of output classes
        self.binary = (nclasses == 1)
        
    """ 
    def forward:
        compute f( c(e(x)), theta(e(x)) ) 
        where e() is encoder, c() is concept, theta() is parametrizer and f() is this class 
    Inputs:
        H:   H(x) vector of concepts (b x k x 1) [TODO: generalize to set maybe?]
        Th:  Theta(x) vector of concept scores (b x k x nclass)
    Return:
        out: predicted task class
    """
    def forward(self, H, Th):
        assert H.size(-2) == Th.size(-2), "Number of concepts in H and Th don't match"
        assert H.size(-1) == 1, "Concept h_i should be scalar, not vector sized"
        assert Th.size(-1) == self.nclasses, "Wrong Theta size"
        combined = torch.bmm(Th.transpose(1,2), H).squeeze(dim=-1)
        
        if self.binary:
            out = F.sigmoid(combined)
        else:
            #for Morpho MNIST and CUB-200-2011
            #out =  F.log_softmax(combined, dim = 1)
            #for BDD-OIA
            out = F.sigmoid(combined)
        return out

    

                                                                                                                                        
