import itertools
import networkx as nx

class UndirectedGraph(nx.Graph):
    
    def __init__(self, ebunch=None):
        super(UndirectedGraph, self).__init__(ebunch)

    def add_node(self, node):
        super(UndirectedGraph, self).add_node(node, weight=None)

    def add_nodes(self, nodes):
        nodes = list(nodes)
        for node in nodes:
            self.add_node(node=node)

    def add_edge(self, nodeA, nodeB):
        super(UndirectedGraph, self).add_edge(nodeA, nodeB, weight=None)

    def add_edges(self, edges):
        edges = list(edges)

        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def is_clique(self, nodes):
        for node1, node2 in itertools.combinations(nodes, 2):
            if not self.has_edge(node1, node2):
                return False
        return True