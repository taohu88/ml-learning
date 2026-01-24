from collections import defaultdict
from itertools import chain

from models import BayesianModel
from util import CPDTable

class InferenceCore(object):

    def __init__(self, model):
        self.model = model
        model.check_model()

        self.variables = model.nodes()

        self.cardinality = {}
        self.factors = defaultdict(list)

        if isinstance(model, BayesianModel):
            for node in model.nodes():
                cpd = model.get_cpds(node)
                if isinstance(cpd, CPDTable):
                    self.cardinality[node] = cpd.variable_card
                    cpd = cpd.to_table()
                for var in cpd.scope():
                    self.factors[var].append(cpd)
        else:
            raise ValueError('Unknown model passed to inference engine')