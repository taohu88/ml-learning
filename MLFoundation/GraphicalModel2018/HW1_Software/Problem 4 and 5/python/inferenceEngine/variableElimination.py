import numpy as np
from inferenceEngine import InferenceCore
from util import factor_product

class VariableElimination(InferenceCore):

    def _variable_elimination(self, variables, operation, evidence=None, elimination_order=None):

        # Dealing with the case when variables is not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            return set(all_factors)

        eliminated_variables = set()
        working_factors = {node: {factor for factor in self.factors[node]}
                           for node in self.factors}

        # Dealing with evidence. Reducing factors over it before VE is run.
        if evidence: #If we have evidence
            for evidence_var in evidence:
                # Iterate through each factor for a given variable
                for factor in working_factors[evidence_var]:
                    # In the factor table remove the probabilities that correspond
                    # to the states that we do not have evidence for
                    factor_reduced = factor.reduce([(evidence_var, evidence[evidence_var])])
                    for var in factor_reduced.scope():
                        working_factors[var].remove(factor)
                        working_factors[var].add(factor_reduced)
                # Delete the evidence's table since we have observed it
                del working_factors[evidence_var]

        # If no elimination order
        if not elimination_order:
            # Just subtract out variables we dont have evidence for
            elimination_order = list(set(self.variables) -
                                     set(variables) -
                                     set(evidence.keys() if evidence else []))
        # Check if we have variables that we wish to find marginals for but are
        # provided as evidence which is incorrect
        elif any(var in elimination_order for var in
                 set(variables).union(set(evidence.keys() if evidence else []))):
            raise ValueError("Elimination order contains variables which are in"
                             " variables or evidence args")

        # Now work on eliminating the variables for which we don't want
        # probabilities for
        for var in elimination_order:
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [factor for factor in working_factors[var]
                       if not set(factor.variables).intersection(eliminated_variables)]
            # Multiply the common probabilities and add new ones to the same table
            # Can be thought of like the product of potentials and probabilities
            phi = factor_product(*factors)
            # Apply probability operation (marginalize)
            phi = getattr(phi, operation)([var], inplace=False)
            # Delete old working factor
            del working_factors[var]
            # Add on new working factor or potentials
            for variable in phi.variables:
                working_factors[variable].add(phi)
            eliminated_variables.add(var)

        # Construct final distributions of variables that are not in the eliminated set
        final_distribution = set()
        for node in working_factors:
            factors = working_factors[node]
            for factor in factors:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add(factor)

        query_var_factor = {}
        # For each variable we quearied
        for query_var in variables:
            # Product of the factors that were not eliminated already
            phi = factor_product(*final_distribution)
            # Marginalize over all remaining variables to yield the proababilities of the variable of interest
            # Also normalize
            query_var_factor[query_var] = phi.marginalize(list(set(variables) -
                                                               set([query_var])),
                                                          inplace=False).normalize(inplace=False)
        return query_var_factor

    def query(self, variables, evidence=None, elimination_order=None):

        return self._variable_elimination(variables, 'marginalize',
                                          evidence=evidence, elimination_order=elimination_order)