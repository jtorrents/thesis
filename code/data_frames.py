import datetime
import os
import csv
from itertools import chain
from collections import defaultdict
from optparse import OptionParser

import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite as bp
#from networkx.algorithms.connectivity.approximation import k_components

from sna.analysis.kcomponents import k_components
from sna import utils

from data import (networks_by_year, networks_by_branches, 
                  get_multigraph, branches, get_peps)
from project import (results_dir, tmp_dir, default_years, default_branches,
                     connectivity_file, survival_file, centrality_file,
                     lifetime_file)

# flatten a nested list
def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

##
## Mobility of devs in the top coonnectivity level network
##
def build_mobility_network(connectivity):
    H = nx.DiGraph()
    networks = networks_by_year()
    skip = next(networks)
    years = range(1992, 2015)
    for year, G in zip(years, networks):
        devs = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        kcomps = connectivity[year]['k_components']
        max_k = max(kcomps)
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        these_devs = set(u for u in set.union(*[v[1] for v in kcomps[max_k]]) if u in devs)
        H.add_node(year, devs=these_devs, number_devs=len(these_devs), total_devs=len(devs))
    for year in years:
        seen = set()
        for future_year in range(year+1, 2015):
            common = H.node[year]['devs'] & H.node[future_year]['devs']
            if common:
                to_add = common - seen
                if to_add:
                    H.add_edge(year, future_year, devs=to_add, weight=len(to_add))
                    seen.update(to_add)
    seen = set()
    for year in years:
        devs = H.node[year]['devs']
        if year == max(years):
            future_devs = set()
        else:
            future_devs = set.union(*[H.node[n]['devs'] for n in range(year+1, 2015)])
        out_devs = devs - future_devs
        if out_devs:
            if year != max(years):
                H.add_node("%s-out" % year, devs=out_devs, number_devs=len(out_devs))
                H.add_edge(year, "%s-out" % year, devs=out_devs, weight=len(out_devs))
        new_devs = devs - seen
        if new_devs:
            H.add_node("%s-in" % year, devs=new_devs, number_devs=len(new_devs))
            H.add_edge("%s-in" % year, year, devs=new_devs, weight=len(new_devs))
        seen.update(devs)
    return H

##
## A helpful summary
##
def summay_results(nets=None, years=None):
    if years is None:
        years = default_years
    if nets is None:
        nets = networks_by_year()
    result = {}
    previous_devs = None
    for year, G in zip(years, nets):
        result[year] = {}
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        files = set(G) - devs
        result[year]['guido_in'] = u'Guido van Rossum' in G
        result[year]['density'] = bp.density(G, devs)
        cc = sorted(nx.connected_components(G), key=len, reverse=True)
        result[year]['cc'] = len(cc[0]) / float(G.order()) if cc else 0
        bcc = sorted(nx.biconnected_components(G), key=len, reverse=True)
        result[year]['bcc'] = len(bcc[0]) / float(G.order()) if bcc else 0
        result[year]['devs'] = len(devs)
        result[year]['files'] = len(files)
        result[year]['py_files'] = len([f for f in files if f.endswith('.py')])
        result[year]['c_files'] = len([f for f in files if f.endswith('.c') 
                                            or f.endswith('.h')])
        result[year]['doc_files'] = len([f for f in files if f.lower().endswith('.txt') 
                                            or f.endswith('.rst') 
                                            or f.endswith('.tex')])
        result[year]['weight'] = sum(nx.degree(G, devs, weight='weight').values())
        result[year]['added'] = sum(nx.degree(G, devs, weight='added').values())
        result[year]['deleted'] = sum(nx.degree(G, devs, weight='deleted').values())
        result[year]['edits'] = sum(nx.degree(G, devs, weight='edits').values())
        result[year]['sq_clustering'] = (sum(nx.square_clustering(G, devs).values()) 
                                            / float(len(devs)))
        if previous_devs is None:
            # First year
            result[year]['new_devs'] = len(devs)
            result[year]['continue_devs'] = 0
            result[year]['lost_devs'] = 0
        else:
            result[year]['new_devs'] = len(devs - previous_devs)
            result[year]['continue_devs'] = len(devs & previous_devs)
            result[year]['lost_devs'] = len(previous_devs - devs)
        previous_devs = devs
    return result
    

def compute_k_components(nets=None, names=None):
    datet = datetime.datetime.today()
    date = datet.strftime("%Y%m%d%H%M")
    if names is None:
        names = default_years
    if nets is None:
        nets = networks_by_year()
    result = {}
    for name, G in zip(names, nets):
        result[name] = {}
        print("Analizing {}".format(name))
        k_comp, k_num = k_components(G)
        result[name]['k_components'] = k_comp
        result[name]['k_num'] = k_num
    fn = 'years' if name == 2014 else 'branches'
    fname = "{0}/k_components_{1}_{2}.pkl".format(results_dir, fn, date)
    utils.write_results_pkl(result, fname)

def compute_layouts(nets=None, names=None):
    datet = datetime.datetime.today()
    date = datet.strftime("%Y%m%d%H%M")
    if names is None:
        names = default_years
    if nets is None:
        nets = networks_by_year()
    result = {}
    for name, G in zip(names, nets):
        # This is likely a pydot bug, nodes cannot have a 'name' attribute 
        for node, data in G.node.items():
            if 'name' in data:
                del data['name']
        result[name] = {}
        print("Analizing {}".format(name))
        result[name]['pos_kk'] = nx.graphviz_layout(G, prog='neato')
        try:
            result[name]['pos_fdp'] = nx.graphviz_layout(G, prog='fdp')
        except:
            result[name]['pos_fdp'] = result[name]['pos_kk']
    fn = 'years' if name == 2014 else 'branches'
    fname = "{0}/layouts_{1}_{2}.pkl".format(results_dir, fn, date)
    utils.write_results_pkl(result, fname)

def devs_by_kcomponent(G, k_components):
    devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
    seen = set()
    result = {}
    for k in sorted(list(k_components), reverse=True):
        nodes= set(n for l in k_components[k]
                            for n in l[1] 
                            if n not in seen
                            and n in devs)
        result[k] = nodes
        seen.update(nodes)
    return result


def compute_centrality(nets=None, names=None):
    datet = datetime.datetime.today()
    date = datet.strftime("%Y%m%d%H%M")
    if names is None:
        names = default_years
    if nets is None:
        nets = networks_by_year()
    result = {}
    for name, G in zip(names, nets):
        result[name] = {}
        print("computing centrality for {}".format(name))
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        result[name]['deg'] = bp.degree_centrality(G, devs)
        try:
            result[name]['bet'] = bp.betweenness_centrality(G, devs)
        except ZeroDivisionError:
            result[name]['bet'] = dict()
        result[name]['clos'] = bp.closeness_centrality(G, devs)
        result[name]['ev'] = nx.eigenvector_centrality_numpy(G)
    fn = 'years' if name == 2014 else 'branches'
    fname = "{0}/bipartite_centrality_{1}_{2}.pkl".format(results_dir, fn, date)
    utils.write_results_pkl(result, fname)

##
## Compute percentage of contributions by connectivity level
##
def contribution_percentage(result=None):
    if result is None:
        result = utils.load_result_pkl(connectivity_file)
    contributions = {}
    max_k = max(flatten([result[year]['k_components'].keys()
                            for year in result]))
    for G in networks_by_year():
        year = G.graph['year']
        contributions[year] = {}
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        total = float(sum(G.degree(devs, weight='weight').values()))
        all_devs = float(len(devs))
        contributions[year]['total'] = (all_devs, 1, total, 1, total/all_devs)
        kcomps = result[year]['k_components']
        for k in range(2, max_k + 1):
            if k not in kcomps:
                contributions[year][k] = (0, 0, 0, 0, 0)
            else:
                nodes_at_k = set.union(*[nodes[1] for nodes in kcomps[k]])
                devs_at_k = nodes_at_k & devs
                if not devs_at_k:
                    print("No developers at level {0} in year {1}".format(k, year))
                    continue
                n_at_k = float(len(devs_at_k))
                contrib_at_k = sum(G.degree(devs_at_k, weight='weight').values())
                contributions[year][k] = (len(devs_at_k),
                                             (len(devs_at_k) / all_devs) * 100,
                                             contrib_at_k,
                                             (contrib_at_k / total) * 100,
                                             contrib_at_k / n_at_k)
    return contributions

##
## Survival analysis
##
def compute_total_contributions(M=None, weight='weight'):
    if M is None:
        M = get_multigraph()
    devs = set(n for n, d in M.nodes(data=True) if d['bipartite']==1)
    return M.degree(devs, weight=weight)

def get_developers_by_years(networks=None):
    if networks is None:
        networks = networks_by_year()
    devs_by_year = {}
    for G in networks:
        year = G.graph['year']
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        devs_by_year[year] = devs
    return devs_by_year

def get_all_remaining_devs(devs_by_year, years):
    return set.union(*[v for k, v in devs_by_year.items() if k in years])

def get_all_developers_top_connectivity(devs_by_year=None, connectivity=None):
    if devs_by_year is None:
        devs_by_year = get_developers_by_years()
    if connectivity is None:
        connectivity = utils.load_result_pkl(connectivity_file)
    all_devs = set.union(*[v for k, v in devs_by_year.items()])
    top_devs = set()
    for year in connectivity:
        kcomponents = connectivity[year]['k_components']
        max_k = max(kcomponents)
        nodes = set.union(*[c[1] for c in kcomponents[max_k]])
        top_devs.update(n for n in nodes if n in all_devs)
    return top_devs

def get_developers_top_connectivity(connectivity, devs):
    max_k = max(connectivity)
    nodes = set.union(*[c[1] for c in connectivity[max_k]])
    return set(n for n in nodes if n in devs)

def get_developers_top_connectivity_by_year(G, year, connectivity=None):
    if connectivity is None:
        connectivity = utils.load_result_pkl(connectivity_file)
    all_devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
    kcomponents = connectivity[year]['k_components']
    max_k = max(kcomponents)
    nodes = set.union(*[c[1] for c in kcomponents[max_k]])
    return set(n for n in nodes if n in all_devs)

def get_developers_top_connectivity_by_year_new(G, year, connectivity=None):
    if connectivity is None:
        connectivity = utils.load_result_pkl(connectivity_file)
    all_devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
    kcomponents = connectivity[year]['k_components']
    max_k = max(kcomponents)
    nodes = set.union(*[c[1] for c in kcomponents[max_k]])
    nodes = nodes | set.union(*[c[1] for c in kcomponents[max_k-1]])
    return set(n for n in nodes if n in all_devs)

def first_seen(devs_by_year):
    births = dict()
    for i, (year, devs) in enumerate(devs_by_year.items()):
        for dev in devs:
            if dev not in births:
                births[dev] = (i, year)
    return births

def compute_lifetime(devs_by_year, years):
    """
    Retruns a dict with each developer as a key and a 
    tuple with 4 values:

    * censorship: 0 if death not observed else 1

    * life time: number of years in the network

    * has been part of the top connectivity level 0 or 1

    * year first seen

    * year last seen

    * Total contributions (lines of code added plus lines removed)

    * Total edits (total number of edits for each file)

    """
    births = first_seen(devs_by_year)
    last_year = max(years)
    last_period = len(years)
    all_devs = set.union(*[v for k, v in devs_by_year.items()])
    top_devs = get_all_developers_top_connectivity(devs_by_year=devs_by_year)
    contributions = compute_total_contributions(weight='weight')
    edits = compute_total_contributions(weight='edits')
    lifetime = dict((dev, (0,
                            last_period - births[dev][0],
                            1 if dev in top_devs else 0,
                            births[dev][1],
                            last_year,
                            contributions[dev],
                            edits[dev],
                            )) for dev in all_devs)
    for i, (year, devs) in enumerate(devs_by_year.items()):
        if year == last_year:
            break
        remaining_devs = get_all_remaining_devs(devs_by_year, years[i+1:])
        for dev in devs:
            if dev not in remaining_devs:
                lifetime[dev] = (1,
                                i + 1 - births[dev][0],
                                1 if dev in top_devs else 0,
                                births[dev][1],
                                year,
                                contributions[dev],
                                edits[dev],
                                )
    return lifetime

def get_lifetime_data_frame(recompute=False, save=False):
    if not recompute and os.path.exists(lifetime_file):
        return pd.read_csv(lifetime_file, index_col=0, encoding='utf8')
    devs_by_year = get_developers_by_years()
    years = default_years
    lifetime = compute_lifetime(devs_by_year, years)
    df = pd.DataFrame(
        data=lifetime.values(),
        index=lifetime.keys(),
        columns=['censored', 'duration', 'top', 'first_year',
                    'last_year', 'contributions', 'edits'],
    )
    df.index.name = 'Developer'
    if save:
        df.to_csv(lifetime_file, encoding='utf8')
    return df


def build_survival_data_frame(fname=survival_file):
    nan = float('nan')
    ids = utils.UniqueIdGenerator()
    connectivity = utils.load_result_pkl(connectivity_file)
    centrality = utils.load_result_pkl(centrality_file)
    peps = [pep for pep in get_peps() if pep.created is not None]
    networks = list(networks_by_year())
    devs = get_developers_by_years(networks=networks)
    skip = networks.pop(0) # skip 1991
    G_start = networks.pop(0) # start with 1992
    devs_start = set(n for n, d in G_start.nodes(data=True) if d['bipartite']==1)
    years = range(1993, 2015)
    with open(fname, 'wb') as f:
        out = csv.writer(f)
        out.writerow([
            'id', 'dev', 'period', 'rstart', 'rstop', 'status',
            'has_written_peps', 'has_written_acc_peps',
            'peps_this_year', 'total_peps',
            'accepted_peps_year', 'total_accepted_peps',
            'biconnected', 'top', 'tenure', 'colaborators',
            'knum', 'aknum', 'clus_sq', 'clus_dot', 'clus_red',
            'degree', 'contributions', 'dcentrality',
            'betweenness', 'closeness',
        ])
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
            peps_this_year = peps_by_developer_that_year(previous_year, peps=peps)
            peps_until_year = peps_by_developer_until_year(previous_year, peps=peps)
            acc_peps_this_year = accepted_peps_by_developer_that_year(previous_year, peps=peps)
            acc_peps_until_year = accepted_peps_by_developer_until_year(previous_year, peps=peps)
            for dev in previous_devs:
                out.writerow([
                    ids[dev], # developer numerical ID
                    dev.encode('utf8'), # developer name
                    i + 1, # period
                    i, # start
                    i + 1, # stop
                    0 if dev in remaining_devs else 1, # status (censored)
                    1 if dev in peps_until_year else 0, # developer has written at least a pep
                    1 if dev in acc_peps_until_year else 0, # developer has written at least an acc. pep
                    peps_this_year[dev] if dev in peps_this_year else 0, # peps written this year
                    peps_until_year[dev] if dev in peps_until_year else 0, # peps written until this year
                    acc_peps_this_year[dev] if dev in acc_peps_this_year else 0, # peps acc. this year
                    acc_peps_until_year[dev] if dev in acc_peps_until_year else 0, # total peps acc.
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


##
## Build data frame for panel regression on developer contributions and 
## zero inflated negative binomial model for PEPs accepted/authored
def compute_tenure_by_year(last_year, networks=None):
    if networks is None:
        networks = networks_by_year()
    result = {}
    seen = set()
    for G in networks:
        year = G.graph['year']
        if year > last_year:
            break
        for node in G:
            if node not in seen:
                result[node] = year
                seen.add(node)
    return dict((node, 1 + last_year - year ) for node, year in result.items())

def normalize(measure):
    max_val = float(max(measure.values()))
    return dict((k, v/max_val) for k, v in measure.items())

def second_order_nbrs(G, u):
    "Second order neighbors"
    return len(set(n for nbr in G[u] for n in G[nbr]) - set([u]))

#
# Analyze PEPs
#
def get_delegates_by_year(year, peps=None):
    if peps is None:
        peps = get_peps()
    delegates = set(flatten(p.delegates for p in peps 
                    if p.delegates and p.created.year == year))
    return {d.first_last for d in delegates}

def count_peps_by_author(peps):
    result = defaultdict(int)
    for pep in peps:
        for author in pep.authors:
            result[author.first_last] += 1
    return result

def peps_by_developer_that_year(year, peps=None):
    if peps is None:
        peps = get_peps()
    peps_year = [pep for pep in peps if pep.created.year == year]
    return count_peps_by_author(peps_year)

def peps_by_developer_until_year(year, peps=None):
    if peps is None:
        peps = get_peps()
    peps_until_year = [pep for pep in peps if pep.created.year <= year]
    return count_peps_by_author(peps_until_year)

def accepted_peps_by_developer_that_year(year, peps=None):
    if peps is None:
        peps = get_peps()
    valid_status = {u'Accepted', u'Final', u'Active'}
    peps_year = [pep for pep in peps if pep.created.year == year]
    accepted_peps = [pep for pep in peps_year if pep.status in valid_status]
    return count_peps_by_author(accepted_peps)

def accepted_peps_by_developer_until_year(year, peps=None):
    if peps is None:
        peps = get_peps()
    valid_status = {u'Accepted', u'Final', u'Active', u'Superseded'}
    peps_until_year = [pep for pep in peps if pep.created.year <= year]
    accepted_peps = [pep for pep in peps_until_year if pep.status in valid_status]
    return count_peps_by_author(accepted_peps)
  
# Write function
def write_developer_contrib_df(fname='data/developer_contributions_df.csv'):
    ids = utils.UniqueIdGenerator()
    peps = [pep for pep in get_peps() if pep.created is not None]
    connectivity = utils.load_result_pkl(connectivity_file)
    centrality = utils.load_result_pkl(centrality_file)
    networks_gen = networks_by_year()
    skip = next(networks_gen)
    networks = list(networks_gen)
    years = range(1992, 2015)
    devs_by_year = get_developers_by_years(networks=networks)
    with open(fname, 'wb') as f:
        out = csv.writer(f)
        out.writerow([
            'id', 'year', 'dev', 'has_written_peps', 'has_written_acc_peps',
            'is_delegate', 'peps_this_year', 'total_peps',
            'accepted_peps_year', 'total_accepted_peps',
            'degree', 'contributions_sc', 'contributions_edits',
            'contributions_added', 'contributions_deleted',
            'collaborators', 'knum', 'aknum', 'top', 'top2',
            'tenure', 'betweenness', 'closeness', 'degree_cent',
            'file_mean_degree', 'clus_sq', 'clus_dot', 'clus_red',
        ])
        for year, G in zip(years, networks):
            print("Analyzing {}".format(G.name))
            bdfl_delegates = get_delegates_by_year(year, peps=peps)
            peps_this_year = peps_by_developer_that_year(year, peps=peps)
            peps_until_year = peps_by_developer_until_year(year, peps=peps)
            acc_peps_this_year = accepted_peps_by_developer_that_year(year, peps=peps)
            acc_peps_until_year = accepted_peps_by_developer_until_year(year, peps=peps)
            top = get_developers_top_connectivity_by_year(G, year,
                                                          connectivity=connectivity)
            top2 = get_developers_top_connectivity_by_year_new(G, year,
                                                               connectivity=connectivity)
            devs = devs_by_year[year]
            tenure = compute_tenure_by_year(year, networks=networks)
            k_num = connectivity[year]['k_num']
            bet = normalize(centrality[year]['bet'])
            clos = normalize(centrality[year]['clos'])
            deg = normalize(centrality[year]['deg'])
            clus_sq = nx.square_clustering(G)
            clus_dot = bp.clustering(G)
            clus_red = bp.node_redundancy(G)
            for dev in devs:
                out.writerow([
                    ids[dev],
                    year,
                    dev.encode('utf8'),
                    1 if dev in peps_until_year else 0, # developer has written at least a pep
                    1 if dev in acc_peps_until_year else 0, # developer has written at least an acc. pep
                    1 if dev in bdfl_delegates else 0, # developer has been BDFL delegate
                    peps_this_year[dev] if dev in peps_this_year else 0, # peps written this year
                    peps_until_year[dev] if dev in peps_until_year else 0, # peps written until this year
                    acc_peps_this_year[dev] if dev in acc_peps_this_year else 0, # peps acc. this year
                    acc_peps_until_year[dev] if dev in acc_peps_until_year else 0, # total peps acc.
                    len(G[dev]), #G.degree(dev, weight=None),
                    G.degree(dev, weight='weight'), # lines of code added plus deleted
                    G.degree(dev, weight='edits'), # number files edit
                    G.degree(dev, weight='added'), # lines of code added
                    G.degree(dev, weight='deleted'), # lines of code removed
                    second_order_nbrs(G, dev), # second order neighbors
                    k_num[dev][0], # k-component number
                    k_num[dev][1], # Average k-component number
                    1 if dev in top else 0, # top connectivity level
                    1 if dev in top2 else 0, # top 2 connectivity level
                    tenure[dev],
                    bet[dev],
                    clos[dev],
                    deg[dev],
                    sum(len(G[n]) for n in G[dev]) / float(len(G[dev])),
                    clus_sq[dev],
                    clus_dot[dev],
                    clus_red[dev],
                ])

##
## Main Function
##

def main():
    parser = OptionParser()
    parser.add_option('-c','--conn_years',
                        action='store_true',
                        dest='connectivity_years',
                        help='Connectivity analysis by year',
                        default=False)
    parser.add_option('-d','--conn_branches',
                        action='store_true',
                        dest='connectivity_branches',
                        help='Connectivity analysis by branch',
                        default=False)
    parser.add_option('-e','--cent_years',
                        action='store_true',
                        dest='centrality_years',
                        help='Centrality analysis by year',
                        default=False)
    parser.add_option('-f','--cent_branches',
                        action='store_true',
                        dest='centrality_branches',
                        help='Centrality analysis by branch',
                        default=False)
    parser.add_option('-l','--layouts_years',
                        action='store_true',
                        dest='layouts_years',
                        help='Compute layouts by year',
                        default=False)
    parser.add_option('-m','--layouts_branches',
                        action='store_true',
                        dest='layouts_branches',
                        help='Compute layouts by branch',
                        default=False)
    parser.add_option('-s','--survival',
                        action='store_true',
                        dest='survival',
                        help='Build and save survival Data Frame',
                        default=False)
    parser.add_option('-z','--contrib',
                        action='store_true',
                        dest='contrib',
                        help='Build and save contribution and PEP authoring DF',
                        default=False)

    options, args = parser.parse_args()

    if options.connectivity_years:
        compute_k_components(nets=networks_by_year(), names=default_years)

    if options.connectivity_branches:
        compute_k_components(nets=networks_by_branches(), names=default_branches)

    if options.centrality_years:
        compute_centrality(nets=networks_by_year(), names=default_years)

    if options.centrality_branches:
        compute_centrality(nets=networks_by_branches(), names=default_branches)

    if options.layouts_years:
        compute_layouts(nets=networks_by_year(), names=default_years)

    if options.layouts_branches:
        compute_layouts(nets=networks_by_branches(), names=default_branches)

    if options.survival:
        build_survival_data_frame(fname=survival_file)

    if options.contrib:
        write_developer_contrib_df(fname='data/developer_contributions_df.csv')

if __name__ == '__main__':
    main()

