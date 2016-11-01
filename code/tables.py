#-*- coding: utf-8 -*-
from __future__ import division

from itertools import chain
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

# Results file
results_file = '../data/developer_contributions_df.csv'

# Define variables
# Variables appear in the same order as written here in the
# descriptive and correlation tables.
dependent_variable = OrderedDict([
    ('contributions_sc', '# of lines of code authored'),
    ('total_accepted_peps', 'Total accepted PEPs'),
])
independent_variables = OrderedDict([
    ('top', 'Top connectivity level'),
    ('knum', r'k-component number'),
])
control_variables = OrderedDict([
    ('degree_cent', 'Degree Centrality'),
    ('tenure', 'Tenure (years)'),
    ('collaborators', 'Collaborators'),
    ('betweenness', 'Betweenness'),
    ('clus_sq', 'Square clustering'),
])

variables = list(
    chain.from_iterable([
        dependent_variable,
        control_variables,
        independent_variables,
        ])
    )

labels = dict(
    chain.from_iterable([
        dependent_variable.items(),
        control_variables.items(),
        independent_variables.items(),
        ])
    )

##
## helpers
##
def almost_zero(x, decimal=7):
    try:
        assert_almost_equal(x,  0, decimal=decimal)
        return True
    except AssertionError:
        return False

def remove_upper(df):
    for i in range(len(df.columns)):
        for j in range(len(df)):
            if i < j:
                df.ix[i, j] = np.nan
    return df
                
##
## Load the results
##
def to_datetime(d):
    if not d:
        return pd.NaT
    return datetime.strptime(str(d), '%Y%m%d')

dataframe = None
def get_dataframe(fname=results_file):
    global dataframe
    if dataframe is None:
        dataframe = pd.read_csv(fname, na_values=['.'])
    return dataframe

##
## Tables
##
# Add numbers to vars as in correlations
def write_descriptives_table(dataframe=None, fname='../tables/table_descriptives.tex'):
    if dataframe is None:
        dataframe = get_dataframe()
    stats = OrderedDict([
        ('count', 'Observations'),
        ('mean', 'Mean'),
        ('std', 'Std. Dev.'),
        ('min', 'Minimum'),
        ('max', 'Maximum'),
    ])
    def formatter(x):
        #print(type(x), x, x % 1)
        if not almost_zero(x % 1, decimal=4):
            return '{:.2f}'.format(x)
        else:
            return '{:,d}'.format(int(x))
    columns = {var:'({}) {}'.format(i, labels[var]) for i, var in enumerate(variables, 1)}
    descriptives = dataframe[variables].describe().rename(columns=columns)
    table = descriptives.transpose()[list(stats)].rename(columns=stats)
    # Fix some values by hand so we don't see thing like 0.0000 as a value
    table['Minimum']['(3) Degree Centrality'] = 0
    with open(fname, 'w') as out:
        out.write('\\begin{table}\n')
        out.write('\caption{Descriptive statistics.}\n')
        out.write('\\begin{center}\n')
        #out.write('\\begin{small}\n')
        #out.write('\\ \n')

        # Write the actual table with pandas
        out.write(table.to_latex(float_format=formatter))

        #out.write('\end{small}\n')
        out.write('\end{center} \n')
        out.write('\end{table}\n')
        #out.write('\ \n')


def write_correlation_table(dataframe=None, fname='../tables/table_correlations.tex'):
    if dataframe is None:
        dataframe = get_dataframe()
    def formatter(x):
        return '{:.3f}'.format(x)
    corr_columns = {var:'({}) {}'.format(i, labels[var]) for i, var in enumerate(variables, 1)}
    correlations = dataframe[variables].corr().rename(columns=corr_columns)
    with open(fname, 'w') as out:
        out.write('\\begin{table}\n')
        out.write('\caption{Correlation matrix.}\n')
        out.write('\\begin{center}\n')
        #out.write('\\begin{small}\n')
        #out.write('\\ \n')

        # Write the actual table with pandas
        # Change column names by numbers
        column_num = {k: i for i, k in enumerate(variables, 1)}
        # Remove redundant values from the upper triangular
        no_upper = remove_upper(correlations.transpose().rename(columns=column_num))
        n = len(no_upper.columns)
        no_upper.drop(n, axis=1, inplace=True)
        out.write(no_upper.to_latex(float_format=formatter).replace('nan', '--'))

        #out.write('\end{small}\n')
        out.write('\end{center} \n')
        out.write('\end{table}\n')
        #out.write('\ \n')

def write_mobility_table(G=None, fname='../tables/table_mobility.tex'):
    if G is None:
        G = get_dataframe()
    def formatter(x):
        #print(type(x), x, x % 1)
        if not almost_zero(x % 1, decimal=4):
            return '{:.1f}'.format(x)
        else:
            return '{:,d}'.format(int(x))
    years = list(range(1992, 2014))
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
    with open(fname, 'w') as out:
        out.write('\\begin{table}\n')
        out.write('\caption{Mobility table.}\n')
        out.write('\\begin{center}\n')
        out.write('\\begin{small}\n')
        #out.write('\\ \n')

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
        dataframe = df.rename(columns=columns)
        out.write(dataframe.to_latex(float_format=formatter))

        out.write('\end{small}\n')
        out.write('\end{center} \n')
        out.write('\end{table}\n')
        #out.write('\ \n')

if __name__ == '__main__':
    write_descriptives_table()
    write_correlation_table()
