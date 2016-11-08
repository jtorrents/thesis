#-*- coding: utf-8 -*-
# Jordi Torrents <jtorrents@milnou.net>
# Functions to compute Small World metrics
from __future__ import division

import argparse
from datetime import datetime
import os
import pickle
import sys

import networkx as nx
from networkx.algorithms import bipartite as bp
from numpy import mean, std

from networks import debian_networks_by_year, python_networks_by_year
from networks import debian_networks_by_release, python_networks_by_branch
from null_model import get_random_networks_2mode
from project import results_dir, tables_dir, python_years, debian_years
from project import python_releases, debian_releases

now = datetime.now().strftime('%Y%m%d%H%M')

##
## pickle results dict
##
def store_result(result, project_name, kind, now=now):
    fname = 'small_world_{}_{}_{}.pkl'.format(project_name, kind, now)
    full_path = os.path.join(results_dir, fname)
    with open(full_path, 'wb') as f:
        pickle.dump(result, f)

##
## Compute CC and APL from random networks
##
def compute_cc_apl_random_networks(G):
    ccs = []
    apls = []
    for Gr in get_random_networks_2mode(G, n=10, r=10):
        ccs.append(bp.robins_alexander_clustering(Gr))
        if not nx.is_connected(Gr):
            Gr = max(nx.connected_component_subgraphs(Gr), key=len)
        apls.append(nx.average_shortest_path_length(Gr))
    return mean(ccs), mean(apls)


##
## Compute Small World Index
##
def compute_bipartite_small_world_index(G):
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    actual_cc = bp.robins_alexander_clustering(G)
    actual_apl = nx.average_shortest_path_length(Gc)
    random_cc, random_apl = compute_cc_apl_random_networks(G)
    result = {}
    result['n'] = G.order()
    result['m'] = G.size()
    if 'python' in G.graph['name'].lower():
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
    try:
        result['swi'] = (actual_cc / random_cc) / (actual_apl / random_apl)
    except ZeroDivisionError:
        result['swi'] = float('nan')
    return result


def compute_small_world_metrics(names_and_networks, project_name, kind):
    result = {}
    for name, G in names_and_networks():
        print('Analyzing {} network by {} for {}'.format(project_name, kind, name))
        result[name] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[name]))
    return result


##
## Convenience function to run interactivelly.
## Not used when this is run as a script.
##
# Python networks
def python_small_world_by_year():
    result = {}
    for year, G in python_networks_by_year():
        print('Analyzing Python network for year {}'.format(year))
        result[year] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[year]))
    return result


def python_small_world_by_release():
    result = {}
    for branch, G in python_networks_by_branch():
        print('Analyzing Python network for branch {}'.format(branch))
        result[branch] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[branch]))
    return result


# Debian networks
def debian_small_world_by_year():
    result = {}
    for year, G in debian_networks_by_year():
        print('Analyzing Debian network for year {}'.format(year))
        result[year] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[year]))
    return result


def debian_small_world_by_release():
    result = {}
    for release, G in debian_networks_by_release():
        print('Analyzing Debian network for release {}'.format(release))
        result[release] = compute_bipartite_small_world_index(G)
        print('    result = {}'.format(result[release]))
    return result


##
## Write table
##
def write_small_world_table(result, project_name, kind):
    fname = 'table_small_world_{}_{}.tex'.format(project_name, kind)
    top = 'Packages' if project_name == 'debian' else 'Files'
    first = 'Years' if kind == 'years' else 'Names'
    if project_name == 'python':
        order = python_years if kind == 'years' else python_releases
    if project_name == 'debian':
        order = debian_years if kind == 'years' else debian_releases
    with open(os.path.join(tables_dir, fname), 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\begin{center}\n")
        #f.write("\\begin{footnotesize}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n")
        f.write("\hline\n")
        header = '%s&Nodes&Developers&%s&Edges&CC&random CC&APL&random APL&SWI ($Q$)\\\\\n'
        f.write(header % (first, top))
        f.write("\hline\n")
        for name in sorted(result, key=order.index):
            row = '{:s}&{:d}&{:d}&{:d}&{:d}&{:.3f}&{:.3f}&{:.1f}&{:.1f}&{:.1f}\\\\\n'
            f.write(row.format(
                        str(name),
                        result[name]['n'],
                        result[name]['developers'],
                        result[name][top.lower()],
                        result[name]['m'],
                        result[name]['actual_cc'],
                        result[name]['random_cc'],
                        result[name]['actual_apl'],
                        result[name]['random_apl'],
                        result[name]['swi'],
                        ))
        f.write("\hline\n")
        f.write("\end{tabular}\n")
        #f.write("\end{footnotesize}\n")
        f.write("\caption{Small world metrics for %s networks.}\n" % project_name)
        f.write("\label{swi_%s}\n" % project_name)
        f.write("\end{center}\n")
        f.write("\end{table}\n")
        f.write("\n")


##
## Main function
##
def main():
    # Print help when we find an error in arguments
    class DefaultHelpParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    #parser = argparse.ArgumentParser()
    parser = DefaultHelpParser()

    group_project = parser.add_mutually_exclusive_group(required=True)
    group_project.add_argument('-d', '--debian', action='store_true',
                               help='Debian networks')
    group_project.add_argument('-p', '--python', action='store_true',
                               help='Python networks')

    group_kind = parser.add_mutually_exclusive_group(required=True)
    group_kind.add_argument('-y', '--years', action='store_true',
                            help='Use year based networks')
    group_kind.add_argument('-r', '--releases', action='store_true',
                            help='Use release based networks')

    args = parser.parse_args()

    if args.debian:
        project_name = 'debian'
        if args.years:
            kind = 'years'
            names_and_networks = debian_networks_by_year
        elif args.releases:
            kind = 'releases'
            names_and_networks = debian_networks_by_release
    elif args.python:
        project_name = 'python'
        if args.years:
            kind = 'years'
            names_and_networks = python_networks_by_year
        elif args.releases:
            kind = 'releases'
            names_and_networks = python_networks_by_branch


    result = compute_small_world_metrics(names_and_networks, project_name, kind)
    store_result(result, project_name, kind)
    write_small_world_table(result, project_name, kind)


if __name__ == '__main__':
    main()
