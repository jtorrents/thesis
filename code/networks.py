# -*- coding: utf-8 -*-
import os

import networkx as nx

from project import multigraph_file, python_years, default_branches
from project import data_dir, debian_years, debian_releases

##
## Build Python networks by year or by branch
##
multigraph = None
def get_python_multigraph():
    global multigraph
    if multigraph is None:
        multigraph = nx.read_graphml(multigraph_file, node_type=str)
    return multigraph


def python_build_graph_by_year(M, year):
    G = nx.Graph()
    G.graph['name'] = "CPython bipartite graph {}".format(year)
    G.graph['year'] = year
    for u, v in set(M.edges()):
        data = M.get_edge_data(u, v).values()
        relevant_data = [d for d in data if d['year']==year]
        if not relevant_data:
            continue
        G.add_edge(u, v,
            edits = len(relevant_data),
            weight = sum([d['weight'] for d in relevant_data]),
            added = sum([d['added'] for d in relevant_data]),
            deleted = sum([d['deleted'] for d in relevant_data]),
        )
    for n in G:
        G.node[n] = M.node[n]
    return G


def python_networks_by_year(M=None, years=python_years):
    if M is None:
        M = get_python_multigraph()
    for year in years:
        yield year, python_build_graph_by_year(M, year)


def python_build_graph_by_branch(M, branch, branches=default_branches):
    if branch not in branches:
        raise ValueError('Not a valid branch % s' % branch)
    G = nx.Graph()
    G.graph['name'] = "CPython bipartite graph {}".format(branch)
    G.graph['branch'] = branch
    for u, v in set(M.edges()):
        data = M.get_edge_data(u, v).values()
        relevant_data = [d for d in data if d['branch']==branch]
        if not relevant_data:
            continue
        G.add_edge(u, v,
            edits=len(relevant_data),
            weight = sum([d['weight'] for d in relevant_data]),
            added = sum([d['added'] for d in relevant_data]),
            deleted = sum([d['deleted'] for d in relevant_data]),
        )
    for n in G:
        G.node[n] = M.node[n]
    return G


def python_networks_by_branch(M=None, branches=default_branches):
    if M is None:
        M = get_python_multigraph()
    for branch in branches:
        yield branch, python_build_graph_by_branch(M, branch)


##
## Build Debian networks by year or by release
##
def debian_networks_by_year(years=debian_years):
    for year in years:
        fname = os.path.join(data_dir, 'debian-UDD-2mode-{}.graphml.gz'.format(year))
        yield year, nx.read_graphml(fname, node_type=str)


def debian_networks_by_release(releases=debian_releases):
    for release in releases:
        fname = os.path.join(data_dir, 'debian-UDD-2mode-{}.graphml.gz'.format(release))
        yield release, nx.read_graphml(fname, node_type=str)
