# Survival anaysis in python
import argparse
from copy import copy
import csv
import os
import sys

import numpy as np
import pandas as pd
# https://github.com/CamDavidsonPilon/lifelines
import lifelines

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

from project import lifetime_file, python_years, tmp_dir, plots_dir
from project import get_structural_cohesion_results
from networks import python_networks_by_year

##
## Helper functions
##
def get_developers_by_years(networks=None):
    if networks is None:
        networks = python_networks_by_year()
    devs_by_year = {}
    for name, G in networks:
        year = G.graph['year']
        devs = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        devs_by_year[year] = devs
    return devs_by_year

def get_all_remaining_devs(devs_by_year, years):
    return set.union(*[v for k, v in devs_by_year.items() if k in years])

def get_developers_top_connectivity(devs_by_year=None):
    if devs_by_year is None:
        devs_by_year = get_developers_by_years()
    if connectivity is None:
        connectivity = get_structural_cohesion_results('python', 'years')
    all_devs = set.union(*[v for k, v in devs_by_year.items()])
    top_devs = set()
    for year in connectivity:
        kcomponents = connectivity[year]['k_components']
        max_k = max(kcomponents)
        nodes = set.union(*[c[1] for c in kcomponents[max_k]])
        top_devs.update(n for n in nodes if n in all_devs)
    return top_devs

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

    """
    births = first_seen(devs_by_year)
    last_year = max(years)
    last_period = len(years)
    all_devs = set.union(*[v for k, v in devs_by_year.items()])
    top_devs = get_developers_top_connectivity(devs_by_year=devs_by_year)
    lifetime = dict((dev, (0, 
                            last_period - births[dev][0], 
                            1 if dev in top_devs else 0,
                            births[dev][1], 
                            last_year,
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
                                )
    return lifetime

def get_lifetime_data_frame(recompute=False):
    if not recompute and os.path.exists(lifetime_file):
        return pd.read_csv(lifetime_file)
    devs_by_year = get_developers_by_years()
    years = python_years
    lifetime = compute_lifetime(devs_by_year, years)
    df = pd.DataFrame(
        data=lifetime.values(),
        index=lifetime.keys(),
        columns=['censored','duration','top','first_year','last_year'], 
    )
    df.index.name = 'Developer'
    return df

def compute_censorship(networks):
    nets = copy(networks)
    G_last = nets.pop()
    survivors = set(n for n, d in G_last.nodes(data=True) if d['bipartite']==1)
    #survivors.union(set(n for n, d in networks[-2].nodes(data=True) if d['bipartite']==1))
    censorship = {}
    seen = set()
    for i, G in enumerate(nets):
        year = G.graph['year']
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        for dev in devs:
            if dev not in seen:
                censorship[dev] = 0 if dev in survivors else 1
                seen.add(dev)
    return censorship


##
## Survival analysts plots
##
def survival_estimation(directory=tmp_dir):
    """ Use the Kaplan-Meier Estimate to estimate the survival function
    
        see: https://github.com/CamDavidsonPilon/lifelines    
    """
    from lifelines.estimation import KaplanMeierFitter

    df = get_lifetime_data_frame(recompute=False)
    # Estimate the survival function for all developers
    T = df['duration']
    C = df['censored']
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=C, label='All developers')
    print("Median survival time for all developers: {} years".format(kmf.median_))
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    kmf.plot(ax=ax, color=color_map(2))
    plt.ylabel('Survival probablility')
    plt.xlabel('Time in years')
    plt.ylim(0,1)
    plt.grid()
    #plt.title("Estimated Survival function for developer activity")
    if directory is None:
        plt.ion()
        plt.show()
    else:
        plt.savefig('{0}/survival_all.png'.format(directory))
        plt.savefig('{0}/survival_all.pdf'.format(directory))
        plt.close()
    # Estimate the survival function by connectivity level
    mtop = df['top'] == 1
    kmf = KaplanMeierFitter()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    kmf.fit(T[mtop], event_observed=C[mtop], label="Top connectivity level")
    print("Median survival time for top developers: {} years".format(kmf.median_))
    kmf.plot(ax=ax, color=color_map(0))
    kmf.fit(T[~mtop], event_observed=C[~mtop], label="Not in the top")
    print("Median survival time for not top developers: {} years".format(kmf.median_))
    kmf.plot(ax=ax, color=color_map(1))
    plt.ylabel('Survival probablility')
    plt.xlabel('Time in years')
    plt.ylim(0,1)
    plt.grid()
    #plt.title("Estimated Survival function for top level connectivity")
    if directory is None:
        plt.ion()
        plt.show()
    else:
        plt.savefig('{0}/survival_top.png'.format(directory))
        plt.savefig('{0}/survival_top.pdf'.format(directory))
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

    parser.add_argument('-p', '--plots', action='store_true',
                        help='Plot Kaplan-Meier survival estimations')


    args = parser.parse_args()

    if args.plots:
        survival_estimation()


if __name__ == '__main__':
    main()
