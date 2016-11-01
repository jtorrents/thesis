import csv

import networkx as nx
from networkx.algorithms import bipartite as bp

from sna import utils

from data import networks_by_year
from project import (data_dir, connectivity_file, 
                    centrality_file, default_years)

nan = float('nan')

def compute_tenure_by_year(last_year):
    result = {}
    seen = set()
    for G in networks_by_year():
        year = G.graph['year']
        if year > last_year:
            break
        for node in G:
            if node not in seen:
                result[node] = year
                seen.add(node)
    return dict((node, last_year - year ) for node, year in result.items())

def second_order_nbrs(G, u):
    "Second order neighbors"
    return len(set(n for nbr in G[u] for n in G[nbr]) - set([u]))

def normalize(measure):
    max_val = float(max(measure.values()))
    return dict((k, v/max_val) for k, v in measure.items())

def get_developers_by_years(networks=None):
    if networks is None:
        networks = networks_by_year()
    devs = {}
    for G in networks:
        these_devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        devs[G.graph['year']] = these_devs
    return devs

def get_all_remaining_devs(devs, years):
    return set.union(*[v for k, v in devs.items() if k in years])

def get_developers_top_connectivity(connectivity, devs):
    max_k = max(connectivity)
    nodes = set.union(*[c[1] for c in connectivity[max_k]])
    return set(n for n in nodes if n in devs)

def build_survival_data_frame(fname="{0}/survival_df.csv".format(data_dir)):
    ids = utils.UniqueIdGenerator()
    connectivity = utils.load_result_pkl(connectivity_file)
    centrality = utils.load_result_pkl(centrality_file)
    networks = list(networks_by_year())
    devs = get_developers_by_years(networks=networks)
    skip = networks.pop(0) # skip 1991
    G_start = networks.pop(0) # start with 1992
    devs_start = set(n for n, d in G_start.nodes(data=True) if d['bipartite']==1)
    years = range(1993, 2014)
    with open(fname, 'wb') as f:
        out = csv.writer(f)
        out.writerow(['id', 'devid',
                        'period', 'rstart', 'rstop', 'status',
                        'biconnected', 'top', 'tenure', 'colaborators',
                        'knum', 'aknum', 'clus_sq', 'clus_dot', 'clus_red',
                        'degree', 'contributions', 'dcentrality',
                        'betweenness', 'closeness'])
        previous_devs = devs_start
        previous_year = 1992
        previous_G = G_start
        for i, (year, G) in enumerate(zip(years, networks)):
            print("processing year {}".format(previous_year))
            clus_sq = nx.square_clustering(previous_G)
            these_devs = devs[year]
            remaining_devs = get_all_remaining_devs(devs, years[i:])
            top_devs = get_developers_top_connectivity(
                connectivity[previous_year]['k_components'], 
                previous_devs)
            tenure = compute_tenure_by_year(previous_year)
            bet = normalize(centrality[previous_year]['bet'])
            clos = normalize(centrality[previous_year]['bet'])
            deg = normalize(centrality[previous_year]['deg'])
            clus_sq = nx.square_clustering(previous_G)
            clus_dot = bp.clustering(previous_G)
            clus_red = bp.node_redundancy(previous_G)
            for dev in previous_devs:
                out.writerow([  ids[dev], # developer numerical ID
                                dev.encode('utf8'), # developer name
                                i + 1, # period
                                i, # start
                                i + 1, # stop
                                0 if dev in remaining_devs else 1, # status (censored)
                                0 if connectivity[previous_year]['k_num'][dev][0] < 2 else 1,#biconnected
                                0 if dev not in top_devs else 1, # member of the top connectivity level
                                tenure[dev], # tenure in years
                                second_order_nbrs(previous_G, dev), # collaborators
                                connectivity[previous_year]['k_num'].get(dev, (nan,nan))[0], # knum
                                connectivity[previous_year]['k_num'].get(dev, (nan,nan))[1], # aknum
                                clus_sq.get(dev, nan),
                                clus_dot.get(dev, nan),
                                clus_red.get(dev, nan),
                                previous_G.degree(dev), # degree
                                previous_G.degree(dev, weight='weight'), # contributions
                                deg.get(dev, nan),
                                bet.get(dev, nan),
                                clos.get(dev, nan),
                            ])
            previous_devs = these_devs
            previous_year = year
            previous_G = G

