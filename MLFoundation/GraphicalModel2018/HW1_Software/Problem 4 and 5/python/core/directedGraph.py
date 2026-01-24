import itertools
import networkx as nx
from core import UndirectedGraph

class DirectedGraph(nx.DiGraph):

    def __init__(self):
        super(DirectedGraph, self).__init__(None)

    def add_node(self, node):
        super(DirectedGraph, self).add_node(node, weight=None)

    def add_nodes(self, nodes):
        nodes = list(nodes)
        for node in nodes:
            self.add_node(node=node)

    def add_edge(self, nodeA, nodeB):
        super(DirectedGraph, self).add_edge(nodeA, nodeB, weight=None)

    def add_edges(self, edges):
        edges = list(edges)
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def moralize(self):
        moral_graph = UndirectedGraph()
        moral_graph.add_edges(self.to_undirected().edges())

        for node in self.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(self.getParents(node), 2))

        return moral_graph

    # ==== Accessors ====
    def getParents(self, node):
        return self.predecessors(node)

    def getChildren(self, node):
        return self.successors(node)

    def getRoots(self):
        return [node for node, in_degree in self.in_degree() if in_degree == 0]

    def getLeaves(self):
        return [node for node, out_degree in self.out_degree() if
                out_degree == 0]
