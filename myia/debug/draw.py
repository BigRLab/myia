"""Visualization tools."""
from collections import defaultdict
from typing import Iterable

import pygraphviz as pgv

from myia.anf_ir import Graph
from myia.ir_utils import dfs


def draw(graphs: Iterable[Graph]) -> pgv.AGraph:
    """Plot a graph using Graphviz."""
    plot = pgv.AGraph()
    subgraphs = defaultdict(list)
    for g in graphs:
        for n in dfs(g.return_):
            if n.graph:
                subgraphs[n.graph].append(n)
            for i in n.inputs:
                plot.add_edge(str(i), str(n))
    for graph, nodes in subgraphs.items():
        plot.add_subgraph(nodes, f"cluster_{str(graph)}")
    return plot
