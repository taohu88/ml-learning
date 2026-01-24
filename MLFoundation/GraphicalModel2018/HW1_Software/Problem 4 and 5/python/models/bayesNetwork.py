from collections import defaultdict
from core import DirectedGraph
from util import CPDTable

class BayesianModel(DirectedGraph):

    def __init__(self):
        super(BayesianModel, self).__init__()
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_cpds(self, *cpds):

         for cpd in cpds:
            if not isinstance(cpd, CPDTable):
                raise ValueError('Only CPDTable can be added.')

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError('CPD defined on variable not in the model', cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def check_model(self):

        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError('No CPD associated with {}'.format(node))
            elif isinstance(cpd, CPDTable):
                evidence = cpd.getEvidence()
                parents = self.getParents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError("CPD associated with {node} doesn't have "
                                     "proper parents associated with it.".format(node=node))
        return True

    def get_cpds(self, node=None):

        if node:
            if node not in self.nodes():
                raise ValueError('Node not present in the Directed Graph')
            for cpd in self.cpds:
                if cpd.variable == node:
                    return cpd
            else:
                return None
        else:
            return self.cpds

