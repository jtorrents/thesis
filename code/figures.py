#-*- coding: utf-8 -*-
# Script to generate figures
from optparse import OptionParser
from collections import defaultdict
from operator import itemgetter
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.sankey import Sankey

import networkx as nx

from sna.plots import structural_cohesion as psc
from sna.analysis import clustering as cl
from sna import utils

from project import plots_dir, connectivity_file, layouts_file
from analysis import contribution_percentage
from data import networks_by_year

##
## Helpers
##
# Log axes labels
def log_10_product_y(x, pos):
    return '%.2f' % (x)
    
def log_10_product_x(x, pos):
    return '%.2f' % (x)

# flatten a nested list
def flatten(listOfLists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(listOfLists)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

##
## Network representation of developer mobility
##
def build_mobility_network(connectivity):
    H = nx.DiGraph()
    networks = networks_by_year()
    skip = next(networks)
    years = range(1992, 2015)
    for year, G in zip(years, networks):
        kcomps = connectivity[year]['k_components']
        max_k = max(kcomps)
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        these_devs = set(u for u in set.union(*[v[1] for v in kcomps[max_k]]) if u in devs)
        H.add_node(year, devs=these_devs, number_devs=len(these_devs))
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


def draw_mobility_network_dot(G, filename='mobility_network_dot_layout'):
    # Convert to Agraph to be able to use dot layout and graphviz nice drawing
    A = nx.to_agraph(G)
    # Graph attributes
    A.graph_attr['label'] = 'Mobility in the top connectivity level'
    A.graph_attr['overlap'] = 'False'  #scale prism
    A.graph_attr['splines'] = 'True'
    # Node attributes
    for node in A.nodes():
        order = int(node.attr['number_devs'])
        node.attr['height'] = '0.20'
        node.attr['width'] = '0.20'
        node.attr['style'] = 'filled'
        #node.attr['fillcolor'] = colors[int(node.attr['k'])]
        # Size of nodes determined by fontsize
        fsize = 5
        if order < 3:
            node.attr['fontsize']= str(1+fsize)
        if order < 5:
            node.attr['fontsize']= str(2+fsize)
        elif order < 7:
            node.attr['fontsize']= str(4+fsize)
        elif order < 10:
            node.attr['fontsize']= str(6+fsize)
        elif order < 13:
            node.attr['fontsize']= str(8+fsize)
        elif order < 15:
            node.attr['fontsize']= str(10+fsize)
        elif order < 20:
            node.attr['fontsize']= str(12+fsize)
        else:
            node.attr['fontsize']= str(14+fsize)
        if 'out' in str(node):
            node.attr['shape'] = 'square'
            node.attr['fillcolor'] = 'red'
            node.attr['label'] = 'OUT'
        elif 'in' in str(node):
            node.attr['shape'] = 'square'
            node.attr['fillcolor'] = 'green'
            node.attr['label'] = 'NEW'
        else:
            node.attr['shape'] = 'circle'
            node.attr['fillcolor'] = 'gray'
            node.attr['label'] = '%s\n%s' % (node, order)
    for edge in A.edges():
        w = int(edge.attr['weight'])
        edge.attr['label'] = str(w)
        thickness = w / 3.
        edge.attr['penwidth'] = str(thickness) if thickness > 1 else '0.5'
        if leaf_nodes_in_edge(edge):
            edge.attr['weight'] = '1'
        if consecutive_nodes(edge):
            edge.attr['weight'] = int(edge.attr['weight']) + 5

    A.layout(prog='dot')
    A.draw(filename+'.png')
    A.draw(filename+'.eps')

def leaf_nodes_in_edge(edge):
    if 'in' in edge[0] or 'in' in edge[1]:
        return True
    if 'out' in edge[0] or 'out' in edge[1]:
        return True
    return False

def consecutive_nodes(edge):
    if not leaf_nodes_in_edge(edge):
        if abs(int(edge[1]) - int(edge[0])) == 1:
            return True
    return False

def draw_mobility_network(G, filename='mobility_network'):
    # Convert to Agraph to be able to use dot layout and graphviz nice drawing
    A = nx.to_agraph(G)
    # Graph attributes
    A.graph_attr['label'] = 'Mobility in the top connectivity level'
    A.graph_attr['overlap'] = 'False'  #scale prism
    A.graph_attr['splines'] = 'True'
    # Node attributes
    for node in A.nodes():
        order = int(node.attr['number_devs'])
        node.attr['height'] = '0.20'
        node.attr['width'] = '0.20'
        node.attr['style'] = 'filled'
        #node.attr['fillcolor'] = colors[int(node.attr['k'])]
        # Size of nodes determined by fontsize
        fsize = 5
        if order < 3:
            node.attr['fontsize']= str(1+fsize)
        if order < 5:
            node.attr['fontsize']= str(2+fsize)
        elif order < 7:
            node.attr['fontsize']= str(4+fsize)
        elif order < 10:
            node.attr['fontsize']= str(6+fsize)
        elif order < 13:
            node.attr['fontsize']= str(8+fsize)
        elif order < 15:
            node.attr['fontsize']= str(10+fsize)
        elif order < 20:
            node.attr['fontsize']= str(12+fsize)
        else:
            node.attr['fontsize']= str(14+fsize)
        if 'out' in str(node):
            node.attr['shape'] = 'square'
            node.attr['fillcolor'] = 'red'
            node.attr['label'] = 'OUT'
            node.attr['pos'] = '400,%s' % (get_y_pos_by_year(node))
        elif 'in' in str(node):
            node.attr['shape'] = 'square'
            node.attr['fillcolor'] = 'green'
            node.attr['label'] = 'NEW'
            node.attr['pos'] = '0,%s' % (get_y_pos_by_year(node))
        else:
            node.attr['shape'] = 'circle'
            node.attr['fillcolor'] = 'gray'
            #node.attr['label'] = '%s' % node
            node.attr['label'] = '%s\n%s' % (node, order)
            node.attr['pos'] = '200,%s' % (get_y_pos_by_year(node))
    for edge in A.edges():
        w = int(edge.attr['weight'])
        edge.attr['xlabel'] = str(w)
        thickness = w / 3.
        edge.attr['penwidth'] = str(thickness) if thickness > 1 else '0.5'
    #A.layout(prog='dot')
    #A.has_layout = True
    A.graph_attr['splines'] = 'True'
    A.graph_attr['sep'] = '0.65'
    #A.layout(prog='neato', args='-n2 -Gsep=.5')
    A.layout(prog='neato', args='-n2')
    A.draw(filename+'.png')
    A.draw(filename+'.eps')

def get_y_pos_by_year(node):
    year = node.split('-')[0]
    pos = {
        '2014': -125,
        '2013': 0,
        '2012': 125,
        '2011': 250,
        '2010': 375,
        '2009': 500,
        '2008': 625,
        '2007': 750,
        '2006': 875,
        '2005': 1000,
        '2004': 1125,
        '2003': 1250,
        '2002': 1375,
        '2001': 1500,
        '2000': 1625,
        '1999': 1725,
        '1998': 1825,
        '1997': 1925,
        '1996': 2025,
        '1995': 2125,
        '1994': 2225,
        '1993': 2325,
        '1992': 2425,
    }
    return pos[year]

##
## Sankey diagrams for developers flow in top connectivity levels
##
def compute_developer_mobility(connectivity):
    networks = networks_by_year()
    skip = next(networks)
    years = range(1992, 2014)
    seen = set()
    flows = []
    last = None
    for year, G in zip(years, networks):
        kcomps = connectivity[year]['k_components']
        max_k = max(kcomps)
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        try:
            these_devs = set(u for u in set.union(*[v[1] for v in kcomps[max_k]]) if u in devs)
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
                          #-len(last - these_devs), # outgoing developers
                          -len(last - these_devs), # outgoing developers
                          -len(last - these_devs), # outgoing developers
                          -len(these_devs)]) # number of developers at time t
            last = these_devs
    return flows

def compute_flows(connectivity):
    networks = networks_by_year()
    skip = next(networks)
    years = range(1992, 2014)
    flows = []
    last = None
    for year, G in zip(years, networks):
        kcomps = connectivity[year]['k_components']
        max_k = max(kcomps)
        devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==1)
        try:
            these_devs = set(u for u in set.union(*[v[1] for v in kcomps[max_k]]) if u in devs)
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

def group_composition_sankey(result, flows=None, fname=None):
    # Data
    if flows is None:
        flows = compute_flows(result)
    years = range(1992, 2014)
    labels = years
    colors = utils.get_colors(len(years))
    # Plot
    fig = plt.figure()
    title = "Composition Flow Diagram for top connectivity"
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title=title)
    sankey = Sankey(ax=ax, scale=0.25, offset=1)
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
        diagram.text.set_fontsize('small')
        for text in diagram.texts:
            text.set_fontsize('small')
    leg = plt.legend(loc='best')
    for t in leg.get_texts():
        t.set_fontsize('small')
    if fname is None:
        plt.ion()
        plt.show()
    else:
        plt.savefig("{0}.eps".format(fname))
        plt.savefig("{0}.png".format(fname))
        plt.close()

##
## Plot functions
##
def plot_bipartite_clustering():
    networks = networks_by_year()
    skip = next(networks)
    years = range(1992, 2014)
    # Square clustering
    sq_top, sq_bot, sq_all = [], [], []
    # Robins and alexander
    ra = []
    # Latapy
    la_top, la_bot, la_all = [], [], []
    # Redundancy
    r_top, r_bot, r_all = [], [], []
    for G in networks:
        year = G.graph['year']
        stop, sbot, sboth = cl.bipartite_square_clustering(G)
        sq_top.append(stop)
        sq_bot.append(sbot)
        sq_all.append(sboth)
        ra.append(cl.bipartite_robins_alexander_clustering(G))
        ltop, lbot, lboth = cl.bipartite_latapy_clustering(G)
        la_top.append(ltop)
        la_bot.append(lbot)
        la_all.append(lboth)
        rtop, rbot, rboth = cl.bipartite_node_redundancy(G)
        r_top.append(rtop)
        r_bot.append(rbot)
        r_all.append(rboth)
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(years, ra, 'o-', c='g')
    ax.plot(years, sq_all, 'o-', c='b')
    ax.plot(years, sq_top, '--', c='b')
    ax.plot(years, r_all, 'o-', c='k')
    ax.plot(years, r_top, '--', c='k')
    ax.set_xlim(min(years), max(years))
    ax.set_ylim(0, 0.35)
    ax.grid(True)
    leg = ax.legend(('Robins-Alexander','Square',' only devs','Redundancy',' only devs'),loc='best')
    for t in leg.get_texts():
        t.set_fontsize('small')
    plt.title('Bipartite Clustering Measures')
    plt.xlabel('Years')
    plt.ylabel('Clustering')
    plt.savefig('bipartite_clustering.png')
    #plt.savefig('sq_clustering.eps')
    plt.close()
 
def plot_assortativity():
    networks = networks_by_year()
    skip = next(networks)
    years = range(1992, 2014)
    asso = []
    for G in networks:
        asso.append(nx.assortativity.degree_assortativity_coefficient(G))
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(years, asso, 'o-', c='g')
    ax.set_xlim(min(years), max(years))
    ax.set_ylim(min(asso)-0.05, max(asso)+0.05)
    ax.grid(True)
    plt.title('Degree Assortativity Evolution')
    plt.xlabel('Years')
    plt.ylabel('Degree Assortativity')
    plt.savefig('deg_asso.png')
    #plt.savefig('deg_asso.eps')
    plt.close()
 
def plot_cohesive_blocks(result):
    for network in result:
        G_k = result[network]['k_components']
        filename = '{0}/cohesive_blocks_{1}'.format(plots_dir, network)
        T = psc.cohesive_blocks(G_k)
        psc.draw_cohesive_blocks(T, filename=filename, prune=False)

def plot_3d_scatter(result, layouts):
    for net in result:
        ### 2 mode plots
        pos_kk = layouts[net]['pos_2m']['kk'] 
        pos_fr = layouts[net]['pos_2m']['fr']
        k_number = result[net]['k_number_2m']
        kk_fname = '{0}/scatter_3d_kk_{1}_2mode'.format(plots_dir, net)
        fr_fname = '{0}/scatter_3d_fr_{1}_2mode'.format(plots_dir, net)
        # XXX Saving twice (png and eps) doesn't work with scatter 3d!
        # no node color on the second file
        psc.plot_scatter_3d_connectivity(k_number, pos_kk, 
                                            filename=kk_fname,
                                            extension='.svg')
        convert_svg_to_eps(kk_fname)
        convert_svg_to_png(kk_fname)
        psc.plot_scatter_3d_connectivity(k_number, pos_fr, 
                                            filename=fr_fname,
                                            extension='.svg')
        convert_svg_to_eps(fr_fname)
        convert_svg_to_png(fr_fname)
        ### 1 mode plots
        pos_kk = layouts[net]['pos_1m']['kk'] 
        pos_fr = layouts[net]['pos_1m']['fr']
        k_number = result[net]['k_number_1m']
        kk_fname = '{0}/scatter_3d_kk_{1}_1mode'.format(plots_dir, net)
        fr_fname = '{0}/scatter_3d_fr_{1}_1mode'.format(plots_dir, net)
        psc.plot_scatter_3d_connectivity(k_number, pos_kk, 
                                            filename=kk_fname,
                                            extension='.svg')
        convert_svg_to_eps(kk_fname)
        convert_svg_to_png(kk_fname)
        psc.plot_scatter_3d_connectivity(k_number, pos_fr, 
                                            filename=fr_fname,
                                            extension='.svg')
        convert_svg_to_eps(fr_fname)
        convert_svg_to_png(fr_fname)

def convert_svg_to_eps(filename):
    from subprocess import check_call
    svg = filename + '.svg'
    eps = filename + '.eps'
    check_call(["convert", svg, eps])

def convert_svg_to_png(filename):
    from subprocess import check_call
    svg = filename + '.svg'
    png = filename + '.png'
    check_call(["convert", svg, png])

def plot_structural_cohesion_barplots(results):
    for net, values in results.items():
        psc.plot_connectivity_barplot(values['k_components_1m'],
                                    values['r_k_components_1m'],
                                    filename='img/barplot_{0}_1mode'.format(net))
        psc.plot_connectivity_barplot(values['k_components_2m'],
                                    values['r_k_components_2m'],
                                    filename='img/barplot_{0}_2mode'.format(net))

##
## Contributions by connectivity level
##

def get_top_level(contributions):
    last_seen = None
    for k, contrib in sorted(contributions.items(), key=itemgetter(0)):
        if not any(contrib) or k == 'total':
            return last_seen
        last_seen = contrib

def connectivity_level_composition(result, 
        fname='{0}/evolution_composition'.format(plots_dir)):
    # Data
    contrib = contribution_percentage(result)
    years = range(1996, 2014)
    max_k = max(flatten([result[year]['k_components'].keys() 
                            for year in result]))
    # percentage of people in the top connectivity level
    top = [get_top_level(contrib[year])[1] for year in years]
    # percentage of people in biconnected components
    bicon = [contrib[year][2][1] for year in years]
    # Plot
    fig = plt.figure()
    ax = plt.subplot(111)
    # percentage of people in the top connectivity level
    ax.plot(years, top, 'o-')
    # percentage of people in biconnected components
    ax.plot(years, bicon, 'o-')
    #ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.grid(True)
    ax.set_ylabel('Percentage of people')
    ax.set_xlabel('Year')
    leg = ax.legend(("top connectivity","biconnected"),loc='best')
    for t in leg.get_texts():
        t.set_fontsize('small')
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    ax.set_title('Evolution of composition of connectivity levels')
    plt.savefig('{0}.png'.format(fname))
    plt.savefig('{0}.eps'.format(fname))
    plt.close()

def contributions_by_connectivity(result, 
        fname='{0}/evolution_contributions'.format(plots_dir)):
    # Data
    contrib = contribution_percentage(result)
    years = range(1996, 2014)
    max_k = max(flatten([result[year]['k_components'].keys() 
                            for year in result]))
    # contributions by people in the top connectivity level
    top = [get_top_level(contrib[year])[3] for year in years]
    # contributions by people in biconnected components
    bicon = [contrib[year][2][3] for year in years]
    # Plot
    fig = plt.figure()
    ax = plt.subplot(111)
    # contributions by people in the top connectivity level
    ax.plot(years, top, 'o-')
    # contributions by people in biconnected components
    ax.plot(years, bicon, 'o-')
    #ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.grid(True)
    ax.set_ylabel('Percentage of contributions')
    ax.set_xlabel('Year')
    leg = ax.legend(("top connectivity","biconnected"),loc='best')
    for t in leg.get_texts():
        t.set_fontsize('small')
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    ax.set_title('Evolution of contributions by connectivity levels')
    plt.savefig('{0}.png'.format(fname))
    plt.savefig('{0}.eps'.format(fname))
    plt.close()


def write_contributions_table(contributions, fname='table_contributions.tex'):
    #contributions = compute_contribution_percentage(result)
    max_k = max(flatten([[k for k in contributions[release].keys() 
                            if isinstance(k, int)] for release in contributions]))
    with open(fname, 'wb') as out:
        out.write("\\begin{landscape}\n")
        out.write("\\begin{table}\n")
        out.write("\\begin{center}\n")
        out.write("\\begin{scriptsize}\n")
        out.write("\\begin{tabular}{|c|c|c|%s}\n"%''.join(['c|c|c|' for i in range(2, max_k + 1)]))
        out.write("\hline\n")
        out.write("&\multicolumn{2}{|c|}{total}%s\\\\ \n"%''.join(["&\multicolumn{3}{|c|}{k=%s}"%k for k in range(2, max_k + 1)]))
        out.write("\hline\n")
        out.write("Networks&\# devs&\# contrib%s\\\\ \n"%''.join(["&\# devs&\% devs&\% contrib" for k in range(2, max_k + 1)]))
        out.write("\hline\n")
        for release in ordered_releases:
            row = [release]
            for i, v in enumerate(contributions[release]['total']):
                if i in [0, 2]:
                    row.append(v)
            for k in range(2, max_k + 1):
                for i, v in enumerate(contributions[release][k]):
                    if i in [0, 1, 3]:
                        row.append(v)
            line = "%s\\\\ \n"%''.join(["&%.0f&%.2f&%.2f" for i in range(2, max_k + 1)])
            str_line = "%s&%.0f&%.0f" + line
            try:
                out.write(str_line % tuple(row))
                out.write("\hline\n")
            except:
                print row
        out.write("\end{tabular}\n")
        out.write("\end{scriptsize}\n")
        out.write("\caption{Contributions table}\n")
        out.write("\label{contrib}\n")
        out.write("\end{center}\n")
        out.write("\end{table}\n")
        out.write("\end{landscape}\n")
        out.write("\n")

def write_contributions_table_slides(contributions, fname='table_slides_contributions.tex'):
    #contributions = compute_contribution_percentage(result)
    max_k = max(flatten([[k for k in contributions[release].keys() 
                            if isinstance(k, int)] for release in contributions]))
    with open(fname, 'wb') as out:
        out.write("\\begin{landscape}\n")
        out.write("\\begin{table}\n")
        out.write("\\begin{center}\n")
        out.write("\\begin{scriptsize}\n")
        out.write("\\begin{tabular}{|c|c|c|%s}\n"%''.join(['c|c|' for i in range(2, max_k + 1)]))
        out.write("\hline\n")
        out.write("&\multicolumn{2}{|c|}{total}%s\\\\ \n"%''.join(["&\multicolumn{2}{|c|}{k=%s}"%k for k in range(2, max_k + 1)]))
        out.write("\hline\n")
        out.write("Networks&\# devs&\# contrib%s\\\\ \n"%''.join(["&\% devs&\% contrib" for k in range(2, max_k + 1)]))
        out.write("\hline\n")
        for release in ordered_releases:
            row = [release]
            for i, v in enumerate(contributions[release]['total']):
                if i in [0, 2]:
                    row.append(v)
            for k in range(2, max_k + 1):
                for i, v in enumerate(contributions[release][k]):
                    if i in [1, 3]:
                        row.append(v)
            line = "%s\\\\ \n"%''.join(["&%.2f&%.2f" for i in range(2, max_k + 1)])
            str_line = "%s&%.0f&%.0f" + line
            try:
                out.write(str_line % tuple(row))
                out.write("\hline\n")
            except:
                print row
        out.write("\end{tabular}\n")
        out.write("\end{scriptsize}\n")
        out.write("\caption{Contributions table}\n")
        out.write("\label{contrib}\n")
        out.write("\end{center}\n")
        out.write("\end{table}\n")
        out.write("\end{landscape}\n")
        out.write("\n")

def write_evolution_table(fname='table_evolution.tex'):
    with open(fname, 'wb') as out:
        out.write("\\begin{table}\n")
        out.write("\\begin{center}\n")
        out.write("\\begin{tabular}{|c|c|c|c|c|}\n")
        out.write("\hline\n")
        out.write("Start date&Release date&Code name&\# developers&\# source packages \\\\ \n")
        out.write("\hline\n")
        for release, G, G1m in get_debian_networks_by_release():
            if release == 'SlinkPotato': continue
            devs = set(n for n, d in G.nodes(data=True) if d['bipartite']==0)
            packs = set(G) - devs
            row = [ releases[release][0].strftime('%Y-%m-%d'),
                    releases[release][1].strftime('%Y-%m-%d'),
                    release,
                    len(devs),
                    len(packs)]
            out.write("%s&%s&%s&%d&%d \\\\ \n" % tuple(row))
        out.write("\hline\n")
        out.write("\end{tabular}\n")
        out.write("\end{center}\n")
        out.write("\end{table}\n")

##
## Main function
##
def main():
    parser = OptionParser()

    parser.add_option('-a','--plot_as', action='store_true',
        dest='plot_as', help='Assortativity evolution', default=False)
    parser.add_option('-b','--plot_bip', action='store_true',
        dest='plot_bip', help='Bipartite clustering evolution', default=False)
    parser.add_option('-c','--plot_cb', action='store_true', 
        dest='plot_cb', help='Plot cohesive blocks', default=False)
    parser.add_option('-d','--plot_s3d', action='store_true', 
        dest='plot_s3d', help='Plot 3d scatter plots', default=False)
    parser.add_option('-j','--plot_contrib', action='store_true',
        dest='plot_contrib', help='contribution percentages', default=False)
    parser.add_option('-k','--plot_knum', action='store_true',
        dest='plot_knum', help='k-number percentages', default=False)
    parser.add_option('-l','--write_contrib', action='store_true', 
        dest='write_contrib', help='Write contributions table (LaTeX)', default=False)
    parser.add_option('-n','--plot_ne', action='store_true', 
        dest='plot_ne', help='Nodes and Edges evolution', default=False)
    parser.add_option('-r','--plot_robustness', action='store_true', 
        dest='plot_robustness', help='Plot robustness for all networks', default=False)
    parser.add_option('-s','--plot_bp', action='store_true', dest='plot_bp',
        help='Plot barplots', default=False)
    parser.add_option('-w','--plot_swi', action='store_true', 
        dest='plot_swi', help='Small World Index evolution', default=False)
    parser.add_option('-z','--plot_sankey', action='store_true', 
        dest='plot_sankey', help='Sankey group composition flow', default=False)

    options, args = parser.parse_args()

    if options.plot_as:
        plot_assortativity_by_release()

    if options.plot_bip:
        plot_bipartite_clustering_by_release()

    if options.plot_cb:
        result = utils.load_result_pkl(results_file)
        plot_cohesive_blocks(result)

    if options.plot_s3d:
        result = utils.load_result_pkl(results_file)
        layouts = utils.load_result_pkl(layouts_file)
        plot_3d_scatter(result, layouts)

    if options.plot_knum:
        result = utils.load_result_pkl(results_file)
        plot_knum_2mode(result)

    if options.plot_contrib:
        result = utils.load_result_pkl(results_file)
        contributions_2mode(result)

    if options.write_contrib:
        result = utils.load_result_pkl(connectivity_file)
        contributions = contribution_percentage(result)
        write_contributions_table(contributions)
        write_contributions_table_slides(contributions)

    if options.plot_ne:
        plot_nodes_and_edges()

    if options.plot_robustness:
        result = utils.load_result_pkl(results_file)
        plot_robustness_random_log(result)
        plot_robustness_random(result)
        plot_robustness_targeted(result)

    if options.plot_bp:
        result = utils.load_result_pkl(results_file)
        plot_structural_cohesion_barplots(result)

    if options.plot_sankey:
        result = utils.load_result_pkl(results_file)
        group_composition_sankey(result, k=5, side='devs', fname='img/sankey_k5')
        group_composition_sankey(result, k=4, side='devs', fname='img/sankey_k4')
        group_composition_sankey(result, k=3, side='devs', fname='img/sankey_k3')
        group_composition_sankey(result, k=2, side='devs', fname='img/sankey_k2')

if __name__ == '__main__':
    main()
