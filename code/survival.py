# Survival anaysis in python
import os
import csv
from itertools import tee, izip
from copy import copy

import numpy as np
import pandas as pd
# https://github.com/CamDavidsonPilon/lifelines
import lifelines

# Set up some better defaults for matplotlib
# Based on https://raw.github.com/cs109/content/master/lec_03_statistical_graphs.ipynb
from matplotlib import rcParams
# colorbrewer2 Dark2 qualitative color table
import brewer2mpl
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors
# we need 10 colors but dark2 has only 8, we almost always will use less than 8
# Thus we append the two first colors for 'Set3', 'Qualitative'
dark2_colors.append((0.5529411764705883, 0.8274509803921568, 0.7803921568627451))
dark2_colors.append((1.0, 1.0, 0.7019607843137254))
#dark2_colors = brewer2mpl.get_map('Set3', 'Qualitative', 10).mpl_colors
rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]

from sna import utils

from project import connectivity_file, survival_file, default_years
from data import networks_by_year

##
## Helper functions
##
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

def get_developers_top_connectivity(devs_by_year=None, connectivity=None):
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
    if not recompute and os.path.exists(survival_file):
        return pd.read_csv(survival_file)
    devs_by_year = get_developers_by_years()
    years = default_years
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

def build_python_survival_df(fname='data/survival_python_df.csv'):
    years = range(1991, 2014)
    networks = list(networks_by_year())
    devs_by_year = get_developers_by_years(networks)
     
