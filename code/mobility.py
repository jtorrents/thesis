#-*- coding: utf-8 -*-
from __future__ import division

import argparse
from collections import OrderedDict
from itertools import chain, cycle
import os
import sys

import networkx as nx
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd

from matplotlib import cm
from matplotlib import rcParams
rcParams['font.family'] = 'sherif'
rcParams['font.serif'] = 'Times'
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = (20, 16)
rcParams['figure.dpi'] = 300
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 18
rcParams['patch.edgecolor'] = 'white'
from matplotlib import pyplot as plt
from matplotlib.sankey import Sankey

from networks import python_networks_by_year, debian_networks_by_year
from project import python_years, debian_years
from project import data_dir, tmp_dir, tables_dir, plots_dir
from project import get_structural_cohesion_results


##
## helpers
##
def almost_zero(x, decimal=4):
    try:
        assert_almost_equal(x,  0, decimal=decimal)
        return True
    except AssertionError:
        return False

##
## Mobility stats
##
def build_mobility_network(names_and_networks, project_name, kind='years'):
    H = nx.DiGraph()
    years = python_years if project_name == 'python' else debian_years
    connectivity = get_structural_cohesion_results(project_name, kind)
    for year, G in names_and_networks():
        if project_name == 'python':
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        else:
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
        kcomps = connectivity[year]['k_components']
        max_k = max(kcomps)
        these_devs = set(u for u in set.union(*[v[1] for v in kcomps[max_k]]) if u in devs)
        H.add_node(year, devs=these_devs, number_devs=len(these_devs), total_devs=len(devs))
    for year in years:
        seen = set()
        for future_year in range(year+1, max(years)+1):
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
            future_devs = set.union(*[H.node[n]['devs'] for n in range(year+1, max(years)+1)])
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
## Mobility table
##
def write_mobility_table(names_and_networks, project_name, directory=tables_dir):
    fname = os.path.join(directory, 'table_mobility_{}.tex'.format(project_name))
    G = build_mobility_network(names_and_networks, project_name, kind='years')
    def formatter(x):
        if not almost_zero(x % 1, decimal=4):
            return '{:.1f}'.format(x)
        else:
            return '{:,d}'.format(int(x))
    years = python_years if project_name == 'python' else debian_years
    columns = OrderedDict([
        ('num_dev', '# Devs top'),
        ('num_dev_pct', '% of all devs'),
        ('new_devs', '# New devs'),
        ('new_devs_pct', '% new devs'),
        ('out_devs', '# Devs out'),
        ('out_devs_pct', '% Devs out'),
        ('came_back', '# Devs back'),
        ('came_back_pct', '% Devs back'),
    ])
    caption = '\caption{Developer mobility in the top connectivity level for the %s project.}\n'
    with open(fname, 'w') as out:
        out.write('\\begin{table}[h]\n')
        out.write(caption % project_name.capitalize())
        out.write('\label{python_mobility_table}\n')
        out.write('\\begin{center}\n')
        #out.write('\\begin{small}\n')

        df = pd.DataFrame(
            index=years,
            columns=[
                'num_dev', 'num_dev_pct',
                'new_devs', 'new_devs_pct',
                'out_devs', 'out_devs_pct',
                'came_back', 'came_back_pct',
            ])
        for year in years:
            total_all_net = G.node[year]['total_devs']
            total = G.node[year]['number_devs']
            node_in = '{}-in'.format(year)
            if node_in in G:
                devs_in = G.node[node_in]['number_devs']
            else:
                devs_in = 0
            node_out = '{}-out'.format(year)
            if node_out in G:
                devs_out = G.node[node_out]['number_devs']
            else:
                devs_out = 0
            back = 0
            for u in G.predecessors(year):
                if u in years:
                    if not u == year-1:
                        back += G.edge[u][year]['weight']
            df.loc[year, 'num_dev'] = total
            df.loc[year, 'num_dev_pct'] = (total /total_all_net) * 100
            df.loc[year, 'new_devs'] = devs_in
            df.loc[year, 'new_devs_pct'] = (devs_in / total) * 100
            df.loc[year, 'out_devs'] = devs_out
            df.loc[year, 'out_devs_pct'] = (devs_out / total) * 100
            df.loc[year, 'came_back'] = back
            df.loc[year, 'came_back_pct'] = (back / total) * 100
        # Write the actual table with pandas
        # Change column names
        #dataframe = df.rename(columns=columns)
        #out.write(dataframe.to_latex(float_format=formatter))
        # Write the actual table
        out.write('\\begin{tabular}{ccccc}\n')
        out.write('\\toprule\n')
        out.write('Years&Top Developers&New Developers&Developers Out&Developers back \\\\ \n')
        out.write('\midrule\n')
        for year in years:
            row = '{}&{} ({:.1f}\%)&{} ({:.1f}\%)&{} ({:.1f}\%)&{} ({:.1f}\%) \\\\ \n'
            out.write(row.format(year,
                df.loc[year, 'num_dev'], df.loc[year, 'num_dev_pct'],
                df.loc[year, 'new_devs'], df.loc[year, 'new_devs_pct'],
                df.loc[year, 'out_devs'], df.loc[year, 'out_devs_pct'],
                df.loc[year, 'came_back'], df.loc[year, 'came_back_pct'],
            ))
        out.write('\\bottomrule\n')
        out.write('\end{tabular}\n')

        #out.write('\end{small}\n')
        out.write('\end{center}\n')
        out.write('\end{table}\n')
        #out.write('\ \n')


##
## Mobility Sankey diagram
## 

def compute_developer_flows(names_and_networks, project_name, kind='years'):
    connectivity = get_structural_cohesion_results(project_name, kind)
    years = python_years if project_name == 'python' else debian_years
    flows = []
    last = None
    for name, G in names_and_networks():
        kcomps = connectivity[name]['k_components']
        max_k = max(kcomps)
        if project_name == 'python':
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        else:
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
        try:
            these_devs = {u for u in set.union(*[v[1] for v in kcomps[max_k]]) if u in devs}
        except TypeError: # Calling set.union without arguments
            if not flows or not any(flows[-1]):
                flows.append([0,0])
            else:
                flows.append([0,0,0,0])
            continue
        if last is None:
            flows.append([len(these_devs), -len(these_devs)])
            last = these_devs
        else:
            flows.append([len(last), # number of developers at time t - 1
                          len(these_devs - last), # incoming new developers
                          -len(last - these_devs), # outgoing developers
                          -len(these_devs)]) # number of developers at time t
            last = these_devs
    return flows


def group_composition_sankey(names_and_networks, project_name,
                             kind='years', directory=plots_dir):
    flows = compute_developer_flows(names_and_networks, project_name)
    years = python_years if project_name == 'python' else debian_years
    labels = years
    color_map = cm.get_cmap('Dark2', len(years))
    colors = [color_map(i) for i in range(len(years))]
    # Plot
    fig = plt.figure(figsize=(20,14))
    title = "Python developer mobility in the top connectivity level"
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title=title)
    sankey = Sankey(ax=ax, scale=0.35, offset=1)
    last = None
    previous_id = 0
    #for i, flow in enumerate(flow for flow in flows if any(flow)):
    for i, flow in enumerate(flows):
        if not any(flow):
            continue
        if last is None: # First group
            sankey.add(flows=flow,
                        fc=colors[i],
                        label=str(labels[i]),
                        orientations=[0, 0])
            last = flow
            previous_id += 1
        else:
            sankey.add(flows=flow,
                        fc=colors[i],
                        label=str(labels[i]),
                        orientations=[0, 1, -1, 0],
                        labels = [None, '', '', ''],
                        #pathlengths=[0.5, 1.75, 1.75, 0.5],
                        pathlengths=[1, 2, 2, 1],
                        prior=previous_id - 1,
                        connect=(3, 0) if len(last) == 4 else (1,0))
            last = flow
            previous_id += 1
    diagrams = sankey.finish()
    for diagram in diagrams:
        diagram.text.set_fontweight('bold')
        #diagram.text.set_fontsize('small')
        #for text in diagram.texts:
        #    text.set_fontsize('small')
    #leg = plt.legend(loc='best')
    leg = ax.legend(bbox_to_anchor=(1, 0.1),
                    ncol=8, fancybox=True, shadow=True)
    #for t in leg.get_texts():
    #    t.set_fontsize('small')
    if directory is None:
        plt.ion()
        plt.show()
    else:
        fname = os.path.join(directory, 'sankey_mobility_{}_{}'.format(project_name, kind))
        plt.savefig("{}.eps".format(fname))
        plt.savefig("{}.png".format(fname))
        plt.close()



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


    group_write = parser.add_mutually_exclusive_group(required=True)
    group_write.add_argument('-t', '--table', action='store_true',
                        help='Write mobility table')
    group_write.add_argument('-s', '--sankey', action='store_true',
                        help='Plot mobility sankey diagram')

    args = parser.parse_args()

    if args.debian:
        raise NotImplementedError
        project_name = 'debian'
        names_and_networks = debian_networks_by_year
    elif args.python:
        project_name = 'python'
        names_and_networks = python_networks_by_year
 
    if args.table:
        write_mobility_table(names_and_networks, project_name)
    elif args.sankey:
        group_composition_sankey(names_and_networks, project_name)


if __name__ == '__main__':
    main()
