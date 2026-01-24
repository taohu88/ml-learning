import numpy as np
import pandas as pd
from itertools import product

class DiscreteTable(object):

    def __init__(self, variables, cardinality, values):

        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=int)
        self.values = values.reshape(self.cardinality)

    def marginalize(self, variables, inplace = True):
        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]

        phi.values = np.sum(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi
    
    def normalize(self, inplace = True):
        phi = self if inplace else self.copy()

        phi.values = phi.values / phi.values.sum()

        if not inplace:
            return phi

    def to_table(self):
        return DiscreteTable(self.variables, self.cardinality, self.values)

    def reduce(self, values):

        phi = DiscreteTable(self.scope(), self.cardinality, self.values)

        var_index_to_del = []
        slice_ = [slice(None)] * len(self.variables)
        for var, state in values:
            # Get index of variables that are given
            var_index = phi.variables.index(var)
            # Add state of the value we have observed
            # Some sort of slice magic, not too sure how this works
            slice_[var_index] = state
            #Add variable index to be deleted from the original
            var_index_to_del.append(var_index)

        var_index_to_keep = sorted(set(range(len(phi.variables))) - set(var_index_to_del))
        # set difference is not gaurenteed to maintain ordering
        phi.variables = [phi.variables[index] for index in var_index_to_keep]
        phi.cardinality = phi.cardinality[var_index_to_keep]
        phi.values = phi.values[tuple(slice_)] #Only get the values with the observed state

        return phi
    
    def product(self, phi1, inplace=True):

        phi = self if inplace else self.copy()
        if isinstance(phi1, (int, float)):
            phi.values *= phi1
        else:
            phi1 = phi1.copy()

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi.values = phi.values[slice_]

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(phi.cardinality, [new_var_card[var] for var in extra_vars])

            # modifying phi1 to add new variables
            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi1.values = phi1.values[slice_]

                phi1.variables.extend(extra_vars)
                # No need to modify cardinality as we don't need it.

            # rearranging the axes of phi1 to match phi
            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], \
                    phi1.variables[axis]
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values * phi1.values

        if not inplace:
            return phi

    def copy(self):
        return DiscreteTable(self.scope(), self.cardinality, self.values)

    # ==== Accessors ====
    def scope(self):
        return self.variables

    def get_cardinality(self, variables):
        return {var: self.cardinality[self.variables.index(var)] for var in variables}

    # ==== Overloads ====
    def __str__(self):
        return self._str(phi_or_p='phi', tablefmt='grid')

    def _str(self, phi_or_p="phi", tablefmt="grid", print_state_names=True):

        string_header = list(map(lambda x: str(x), self.scope()))
        string_header.append('{phi_or_p}({variables})'.format(phi_or_p=phi_or_p,
                                                              variables=','.join(string_header)))

        value_index = 0
        factor_table = []
        for prob in product(*[range(card) for card in self.cardinality]):
            # Generate list of probabilites
            prob_list = ["{s}_{d}".format(s=list(self.variables)[i], d=prob[i])
                            for i in range(len(self.variables))]

            prob_list.append(self.values.ravel()[value_index])
            factor_table.append(prob_list)
            value_index += 1

        np_table = np.concatenate((np.expand_dims(string_header,0), factor_table), axis=0)

        return pd.DataFrame(np_table).to_string()
    
    def __mul__(self, other):
        return self.product(other, inplace=False)