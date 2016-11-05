#-*- coding: utf-8 -*-
# Jordi Torrents <jtorrents@milnou.net>
# Functions to generate random networks
from __future__ import division

from datetime import datetime
import os
import pickle
import random

import networkx as nx
from networkx.algorithms import bipartite as bp
from numpy import mean, std

from networks import debian_networks_by_year, python_networks_by_year
from networks import debian_networks_by_release, python_networks_by_branch
from project import results_dir

now = datetime.now().strftime('%Y%m%d%H%M')

##
## pickle results dict
##
def store_result(result, name, now=now):
    fname = 'small_world_{}_{}.pkl'.format(name, now)
    full_path = os.path.join(results_dir, fname)
    with open(full_path, 'wb') as f:
        pickle.dump(result, f)

##
## Bipartite Configuration Model Random Networks
##
def generate_random_configuration_2mode(G):
    top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    bot_nodes = set(G) - top_nodes
    aseq = [d for n, d in G.degree(top_nodes)]
    bseq = [d for n, d in G.degree(bot_nodes)]
    return bp.configuration_model(aseq, bseq, create_using=nx.Graph())


def generate_random_2mode(G):
    top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    bot_nodes = set(G) - top_nodes
    size = G.size()
    density = bp.density(G, top_nodes)
    #return bp.gnmk_random_graph(len(top_nodes), len(bot_nodes), size)
    return bp.random_graph(len(top_nodes), len(bot_nodes), density)


##
## Compute CC and APL from random networks
##
def compute_cc_apl_random_networks(G):
    Gr = generate_random_2mode(G)
    cc = bp.average_clustering(Gr)
    if not nx.is_connected(Gr):
        Gr = max(nx.connected_component_subgraphs(Gr), key=len)
    apl = nx.average_shortest_path_length(Gr)
    return cc, apl


##
## Compute Small World Index
##
def compute_bipartite_small_world_index(G):
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    actual_cc = bp.average_clustering(G)
    actual_apl = nx.average_shortest_path_length(Gc)
    random_cc, random_apl = compute_cc_apl_random_networks(G)
    result = {}
    result['n'] = G.order()
    result['m'] = G.size()
    if 'Python' in G.graph['name']:
        devs = len({n for n, d in G.nodes(data=True) if d['bipartite'] == 1})
        result['developers'] = devs
        result['files'] = G.order() - devs
    else:
        devs = len({n for n, d in G.nodes(data=True) if d['bipartite'] == 0})
        result['developers'] = devs
        result['packages'] = G.order() - devs
    result['actual_cc'] = actual_cc
    result['actual_apl'] = actual_apl
    result['random_cc'] = random_cc
    result['random_apl'] = random_apl
    result['swi'] = (actual_cc / random_cc) / (actual_apl / random_apl)
    return result

##
## Python networks
##
def python_small_world_by_year():
    result = {}
    for year, G in python_networks_by_year():
        print('Analyzing Python network for year {}'.format(year))
        result[year] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[year]))
    return result


def python_small_world():
    result = {}
    for branch, G in python_networks_by_branch():
        print('Analyzing Python network for branch {}'.format(branch))
        result[branch] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[branch]))
    return result


##
## Debian networks
##
def debian_small_world_by_year():
    result = {}
    for year, G in debian_networks_by_year():
        print('Analyzing Debian network for year {}'.format(year))
        result[year] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[year]))
    return result


def debian_small_world():
    result = {}
    for release, G in debian_networks_by_release():
        print('Analyzing Debian network for release {}'.format(release))
        result[release] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[release]))
    return result


if __name__ == '__main__':
    python_result = python_small_world_by_year()
    store_result(python_result, 'python')
    debian_result = debian_small_world_by_year()
    store_result(debian_result, 'debian')
