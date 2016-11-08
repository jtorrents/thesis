#-*- coding: utf-8 -*-
# Jordi Torrents <jtorrents@milnou.net>
# Functions to generate random networks
import random

import networkx as nx
from networkx.algorithms import bipartite as bp

##
## Random Networks for Null models
##
# Bipartite Configuration Model random networks
def generate_random_configuration_2mode(G):
    top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    bot_nodes = set(G) - top_nodes
    aseq = [d for n, d in G.degree(top_nodes)]
    bseq = [d for n, d in G.degree(bot_nodes)]
    return bp.configuration_model(aseq, bseq, create_using=nx.Graph())


# Bipartite Erdos-Renyi Model random networks
def generate_random_2mode(G):
    top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    bot_nodes = set(G) - top_nodes
    size = G.size()
    density = bp.density(G, top_nodes)
    #return bp.gnmk_random_graph(len(top_nodes), len(bot_nodes), size)
    return bp.random_graph(len(top_nodes), len(bot_nodes), density)


def get_random_networks_2mode(G, n=10, r=10):
    """ Generator of random 2 mode networks

    G: NetworkX bipartite graph

    n: Total number of random networks to yield

    r: number of networks generated for each network yielded

    The total number of networks generated (ie universe) is n * r
    """
    for i in range(n):
        for j in range(r):
            random_net = generate_random_configuration_2mode(G)
            if random.random() <= 1. / (r - j):
                yield random_net
                break

