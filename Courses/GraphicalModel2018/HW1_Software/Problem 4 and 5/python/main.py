# ==================================================
# Homework 1, Problem 4 and 5
# Center for Informatics and Computational Sciences
# University of Notre Dame
# ==================================================

import numpy as np

from util.cpd import CPDTable
from models.bayesNetwork import BayesianModel
from inferenceEngine import VariableElimination

if __name__ == '__main__':
    
    # Construct Bayes PGM
    model = BayesianModel()
    # Add edges (and nodes)
    model.add_edges([('cl', 'sp'), ('cl', 'rn'), ('sp', 'wg'), ('rn', 'wg')])
    print('Roots:', model.getRoots())
    print('Leaves:', model.getLeaves())

    # Construct CPD Tables for each node
    cpd_cl = CPDTable('cl', 2, values=[[0.5], [0.5]])
    cpd_sp = CPDTable('sp', 2, values=[[0.5,0.9], [0.5,0.1]],
                        evidence=['cl'], evidence_card=[2])
    cpd_rn = CPDTable('rn', 2, values=[[0.8,0.2], [0.2,0.8]],
                        evidence=['cl'], evidence_card=[2])
    cpd_wg = CPDTable('wg', 2, values=[[1.0,0.1,0.1,0.01], [0.0,0.9,0.9,0.99]],
                        evidence=['sp', 'rn'], evidence_card=[2,2])
    print('==== Cloudy CPD ====')
    print(cpd_cl)
    print('==== Sprinkler CPD ====')
    print(cpd_sp)
    print('==== Rain CPD ====')
    print(cpd_rn)
    print('==== Wet Grass CPD ====')
    print(cpd_wg)

    # Add CPDs to the graph and check for inconsistancies
    model.add_cpds(cpd_cl, cpd_sp, cpd_rn, cpd_wg)
    model.check_model()
    model.moralize() # Moralize and convert to undirected graph

    # Now infer
    infer = VariableElimination(model)
    result = infer.query(['sp'], evidence={'wg': 1})
    
    print('==== P(S|WG=1) ====')
    print(result['sp'])




    
