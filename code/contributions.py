#-*- coding: utf-8 -*-
# Jordi Torrents <jtorrents@milnou.net>
# Functions to compute contributions related metrics
from __future__ import division

import argparse
from collections import defaultdict
from operator import itemgetter
import os
import sys

from matplotlib import cm
color_map = cm.get_cmap('Dark2', 8)
from matplotlib import rcParams
rcParams['font.family'] = 'sherif'
rcParams['font.serif'] = 'Times'
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = (10, 8)
rcParams['figure.dpi'] = 300
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 16
rcParams['patch.edgecolor'] = 'white'
from matplotlib import pyplot as plt

from networks import debian_networks_by_year, python_networks_by_year
from project import plots_dir, tmp_dir
from project import python_years, debian_years
from project import get_structural_cohesion_results
from project import get_structural_cohesion_null_model_results
from utils import flatten

##
## Get structural cohesion analysis 
##
def structural_cohesion_results(project_name, kind):
    actual = get_structural_cohesion_results(project_name, kind)
    null = get_structural_cohesion_null_model_results(project_name, kind)
    return actual, null

##
## Contribution percentage by connectivity level
##

def compute_contribution_percentage(names_and_networks, connectivity, project_name, kind):
    contributions = {}
    for name, G in names_and_networks():
        contributions[name] = {}
        if project_name == 'debian':
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
        else:
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        total = sum(deg for n, deg in G.degree(devs, weight='weight'))
        all_devs = len(devs)
        contributions[name]['total'] = (all_devs, 1, total, 1, total/all_devs)
        kcomps = connectivity[name]['k_components']
        max_k = max(kcomps.keys())
        for k in range(2, max_k + 1):
            if k not in kcomps:
                contributions[name][k] = (0, 0, 0, 0, 0)
            else:
                nodes_at_k = set.union(*[nodes[1] for nodes in kcomps[k]])
                devs_at_k = nodes_at_k & devs
                n_at_k = len(devs_at_k)
                contrib_at_k = sum(deg for n, deg in G.degree(devs_at_k, weight='weight'))
                contributions[name][k] = (
                    len(devs_at_k),
                    (len(devs_at_k) / all_devs) * 100,
                    contrib_at_k,
                    (contrib_at_k / total) * 100,
                    contrib_at_k / n_at_k if n_at_k != 0 else float('nan'),
                )
    return contributions

def compute_contribution_percentage_top(names_and_networks, connectivity, project_name, kind):
    contributions = {}
    for name, G in names_and_networks():
        contributions[name] = {}
        if project_name == 'debian':
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
        else:
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        total = sum(deg for n, deg in G.degree(devs, weight='weight'))
        all_devs = len(devs)
        contributions[name]['total'] = (all_devs, 1, total, 1, total/all_devs)
        kcomps = connectivity[name]['k_components']
        # Compute contributions by developers at the in the bicomponent
        nodes_at_2 = set.union(*[nodes[1] for nodes in kcomps[2]])
        devs_at_2 = nodes_at_2 & devs
        n_at_2 = len(devs_at_2)
        contrib_at_2 = sum(deg for n, deg in G.degree(devs_at_2, weight='weight'))
        contributions[name]['bicomponent'] = (
            len(devs_at_2),
            (len(devs_at_2) / all_devs) * 100,
            contrib_at_2,
            (contrib_at_2 / total) * 100,
            contrib_at_2 / n_at_2,
        )
        # And then for all connectivity levels for k >= 3 in the case of Debian
        # or only the top connectivity level in the case of Pythom (because)
        # there are many developers in python for k >= 3
        max_k = max(connectivity[name]['k_components'].keys())
        min_k = 3 if project_name == 'debian' else max_k
        nodes_at_top = set()
        devs_at_top = set()
        n_at_top = 0
        contrib_at_top = 0
        for k in range(min_k, max_k + 1):
            nodes_at_k = set.union(*[nodes[1] for nodes in kcomps[k]])
            devs_at_k = nodes_at_k & devs
            nodes_at_top |= nodes_at_k
            devs_at_top |= devs_at_k
            n_at_top += len(devs_at_k)
            contrib_at_top += sum(deg for n, deg in G.degree(devs_at_k, weight='weight'))
        contributions[name]['top'] = (
            len(devs_at_top),
            (len(devs_at_top) / all_devs) * 100,
            contrib_at_top,
            (contrib_at_top / total) * 100,
            contrib_at_top / n_at_top if n_at_top != 0 else float('nan'),
        )
    return contributions

##
## Compute developers by connectivity level
##
def compute_developers_percentage_top(names_and_networks, connectivity, project_name, kind):
    developers = {}
    for name, G in names_and_networks():
        developers[name] = {}
        if project_name == 'debian':
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
        else:
            devs = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        total = len(devs)
        developers[name]['total'] = total
        kcomps = connectivity[name]['k_components']
        # Compute percentage developers in the bicomponent
        nodes_at_2 = set.union(*[nodes[1] for nodes in kcomps[2]])
        devs_at_2 = nodes_at_2 & devs
        developers[name]['bicomponent'] = (len(devs_at_2) / total) * 100
        # And then developers for all connectivity levels for k >= 3
        # in the case of Debian or only the top connectivity level in
        # the case of Pythom because there are many developers in
        # python for k >= 3
        max_k = max(connectivity[name]['k_components'].keys())
        min_k = 3 if project_name == 'debian' else max_k
        nodes_at_top = set()
        devs_at_top = set()
        for k in range(min_k, max_k + 1):
            nodes_at_k = set.union(*[nodes[1] for nodes in kcomps[k]])
            devs_at_k = nodes_at_k & devs
            nodes_at_top |= nodes_at_k
            devs_at_top |= devs_at_k
        developers[name]['top'] = (len(devs_at_top) / total) * 100
    return developers



##
## Plot percentage of developers by connectivity level
##
def plot_developers_by_connectivity(names_and_networks, connectivity,
                                    project_name, kind, directory=tmp_dir):
    # Data
    dates = python_years if project_name == 'python' else debian_years
    developers = compute_developers_percentage_top(names_and_networks, connectivity,
                                                   project_name, kind)
    dates, top, bicomp = zip(*[(date, d['top'], d['bicomponent']-d['top'])
                                for date, d in sorted(developers.items())])
    # Plot
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.stackplot(dates, top, bicomp, colors=(color_map(0), color_map(1)))
    ax.set_ylim(0, 100)
    ax.set_xlim(1999, 2013 if project_name == 'python' else 2011)
    ax.grid(True)
    ax.set_ylabel('Percentage of developers')
    if project_name == 'debian':
        leg = ax.legend((r'$k >= 3$', r'bicomponent ($k=2$)'), 
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1),
                        ncol=2, fancybox=True, shadow=True)
    else:
        leg = ax.legend(('top connectivity level', r'bicomponent ($k=2$)'),
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1),
                        ncol=2, fancybox=True, shadow=True)
    #for t in leg.get_texts():
    #    t.set_fontsize('small')
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    ax.set_title('Developer percentage by connectivity level')
    if directory is None:
        plt.ion()
        plt.show()
    else:
        name = 'evolution_developers_{}_{}'.format(project_name, kind)
        fname = os.path.join(directory, name)
        plt.savefig('{}.png'.format(fname))
        plt.savefig('{}.eps'.format(fname))
        plt.close()


def plot_contributions_by_connectivity(names_and_networks, connectivity,
                                       project_name, kind, directory=tmp_dir):
    # Data
    contrib = compute_contribution_percentage_top(
                names_and_networks, connectivity, project_name, kind)
    dates, top, bicomp = zip(*[(date, d['top'][3], d['bicomponent'][3]-d['top'][3])
                                for date, d in sorted(contrib.items())])
    # Plot
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.stackplot(dates, top, bicomp, colors=(color_map(0), color_map(1)))
    ax.set_ylim(0, 100)
    ax.set_xlim(1999, 2013 if project_name == 'python' else 2011)
    ax.grid(True)
    ax.set_ylabel('Contribution percentage of developers')
    if project_name == 'debian':
        leg = ax.legend((r'$k >= 3$', r'bicomponent ($k=2$)'), 
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1),
                        ncol=2, fancybox=True, shadow=True)
    else:
        leg = ax.legend(('top connectivity level', r'bicomponent ($k=2$)'),
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1),
                        ncol=2, fancybox=True, shadow=True)
    #for t in leg.get_texts():
    #    t.set_fontsize('small')
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    ax.set_title('Developer contribution by connectivity level')
    if directory is None:
        plt.ion()
        plt.show()
    else:
        name = 'evolution_connectivity_{}_{}'.format(project_name, kind)
        fname = os.path.join(directory, name)
        plt.savefig('{}.png'.format(fname))
        plt.savefig('{}.eps'.format(fname))
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

    args = parser.parse_args()

    if args.debian:
        project_name = 'debian'
        kind = 'years'
        names_and_networks = debian_networks_by_year
    elif args.python:
        project_name = 'python'
        kind = 'years'
        names_and_networks = python_networks_by_year

    connectivity = get_structural_cohesion_results(project_name, kind)
    plot_developers_by_connectivity(names_and_networks, connectivity, project_name, kind)
    plot_contributions_by_connectivity(names_and_networks, connectivity, project_name, kind)


if __name__ == '__main__':
    main()
