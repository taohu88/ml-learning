import numbers
from itertools import product
import pandas as pd
import numpy as np

from discreteTable import DiscreteTable

class CPDTable(DiscreteTable):

    def __init__(self, variable, variable_card, values,
                 evidence=None, evidence_card=None):

        self.variable = variable
        self.variable_card = None

        variables = [variable]

        if not isinstance(variable_card, numbers.Integral):
            raise TypeError("Event cardinality must be an integer")
        self.variable_card = variable_card

        cardinality = [variable_card]
        if evidence_card is not None:
            if isinstance(evidence_card, numbers.Real):
                raise TypeError("Evidence card must be a list of numbers")
            cardinality.extend(evidence_card)

        if evidence is not None:
            variables.extend(evidence)
            if not len(evidence_card) == len(evidence):
                raise ValueError("Length of evidence_card doesn't match length of evidence")

        values = np.array(values)
        if values.ndim != 2:
            raise TypeError("Values must be a 2D list/array")

        super(CPDTable, self).__init__(variables, cardinality, values.flatten('C'))

    def get_values(self):
        if self.variable in self.variables:
            return self.values.reshape(self.cardinality[0], np.prod(self.cardinality[1:]))
        else:
            return self.values.reshape(1, np.prod(self.cardinality))
    
    def getEvidence(self):
        return self.variables[:0:-1]
    
    def __str__(self):
        return self._make_table_str(tablefmt="grid")

    def _make_table_str(self, tablefmt="fancy_grid", print_state_names=True):
        headers_list = []
        # build column headers

        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))
            for i in range(len(evidence_card)):
                column_header = [str(evidence[i])] + ['{s}_{d}'.format(
                    s=evidence[i], d=d) for d in col_indexes.T[i]]
                headers_list.append(column_header)

        # Build row headers
        variable_array = [['{s}_{d}'.format(s=self.variable, d=i) for i in range(self.variable_card)]]
        # Stack with data
        labeled_rows = np.hstack((np.array(variable_array).T, self.get_values())).tolist()
        
        if(np.array(labeled_rows).ndim == 1):
            labeled_rows = np.expand_dims(labeled_rows, 0)

        # If we have evidence headers
        if(len(headers_list) > 0):
            # Concat headers and rows
            np_table = np.concatenate((headers_list,labeled_rows), axis=0)
        else:
            np_table = labeled_rows
        
        return pd.DataFrame(np_table).to_string()
