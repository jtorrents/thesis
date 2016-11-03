#! /usr/bin/env python
#-*- coding: utf-8 -*-
# Jordi Torrents <jtorrents@milnou.net>
from __future__ import division

import os
import random
from math import log
from copy import deepcopy
from operator import itemgetter

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from numpy import mean, std
import matplotlib.pyplot as P
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

plots_dir = os.path.join(os.pardir, 'tmp')

def rel_size_giant_bicomponent(G):
    G_b = max(nx.biconnected_components(G), key=len)
    return len(G_b)/len(G)

def get_swi(G):
    # Erdös Renyi Random Model
    k = (2*G.size())/float(G.order())
    n = G.order()
    r_apl = log(n)/log(k)
    r_cc = k/n
    cc = nx.average_clustering(G)
    apl = nx.average_shortest_path_length(G)
    return (cc/r_cc)/(apl/r_apl)

def draw_graph(G, filename, title, prog='neato', size=8, node_size=300, labels=False, pos=None):
    filename = os.path.join(plots_dir, filename)
    P.figure(figsize=(size,size))
    P.title(title)
    cc = nx.average_clustering(G)
    apl = nx.average_shortest_path_length(G)
    n = G.order()
    m = G.size()
    swi = get_swi(G)
    sgbc = rel_size_giant_bicomponent(G)
    if not pos:
        pos = graphviz_layout(G, prog=prog)
    nx.draw(G, pos, node_color='red', node_size=node_size, with_labels=labels)
    ax = P.gca()
    msg = "N=%d, M=%d, CC=%.2f, APL=%.2f, SWI=%.2f, nodes in GBC=%.2f" % (n,m,cc,apl,swi,sgbc)
    P.text(0.5, 0.01, msg, horizontalalignment='center', verticalalignment='center',
           transform = ax.transAxes, fontsize=14)
    P.axis('off')
    P.savefig(filename+".png")
    P.savefig(filename+".eps")
    P.close()

def component_sensivity(g,mode='target', relative=False):
    total = g.order()
    ngc = len(max(nx.connected_components(g), key=len))
    if mode == 'target':
        graph = deepcopy(g)
        result = []
        result.append((0,1))
        deg = dict(graph.degree())
        for i,(n,v) in enumerate(sorted(deg.items(),key=itemgetter(1),reverse=True)):
            graph.remove_node(n)
            if graph.size() == 0:
                break
            if relative:
                result.append((
                    (i+1) / total,
                    len(max(nx.connected_components(graph), key=len)) / graph.order()
                ))
            else:
                result.append((
                    (i+1)/total,
                    len(max(nx.connected_components(graph), key=len)) / ngc
                ))
        return result[:int(total*0.8)]
    elif mode == 'random':
        res = []
        for j in range(10):
            graph = deepcopy(g)
            r = []
            nodes = list(graph.nodes())
            for i in range(len(nodes)):
                n = random.choice(nodes)
                graph.remove_node(n)
                nodes.remove(n)
                if graph.size() == 0:
                    break
                if relative:
                    r.append(len(max(nx.connected_components(graph), key=len)) / graph.order())
                else:
                    r.append(len(max(nx.connected_components(graph), key=len)) / ngc)
            res.append(r)
        result= []
        result.append((0,1,0))
        #for i in range(int(total*0.8)):
        for i in range(min([len(r) for r in res])):
            values = [x[i] for x in res]
            result.append(((i+1)/total, mean(values), std(values)))
        return result

def plot_csen(graph,model,n=25):
    # Targeted removal
    result_t = component_sensivity(graph, mode='target')
    x_t = [res[0] for res in result_t]
    y_t = [res[1] for res in result_t]
    # Random removal
    result_r = component_sensivity(graph, mode='random')
    x_r = [res[0] for res in result_r]
    y_r = [res[1] for res in result_r]
    sd = [res[2] for res in result_r]
    # plot the graphs
    P.figure(figsize=(8,8))
    ax = P.subplot(111)
    ax.plot(x_t, y_t, 'o-', c='r')
    ax.plot(x_r, y_r, 'x-', c='b')
    ax.errorbar(x_r, y_r, yerr=sd, fmt='bx')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    leg = ax.legend(('Targeted removal','Random removal'),loc='upper right')
    for t in leg.get_texts():
        t.set_fontsize('small')
    P.title('Robustness %s'%model)
    P.xlabel('Percentage nodes removed')
    P.ylabel('Relative size of Giant Component')
    P.savefig('../tmp/csen_%s.png'%model.replace(' ','_'))
    P.savefig('../tmp/csen_%s.eps'%model.replace(' ','_'))
    P.close()

def plot_csen_log(graph, model, n=25):
    # Targeted removal
    result_t = component_sensivity(graph, mode='target', relative=True)
    x_t = [res[0] for res in result_t]
    y_t = [res[1] for res in result_t]
    # Random removal
    result_r = component_sensivity(graph, mode='random', relative=True)
    x_r = [res[0] for res in result_r]
    y_r = [res[1] for res in result_r]
    sd = [res[2] for res in result_r]
    # plot the graphs
    P.figure(figsize=(8,8))
    def log_10_product_y(x, pos):
        return '%.2f' % (x)
    def log_10_product_x(x, pos):
        #return '%1i' % (x)
        return '%.2f' % (x)
    ax = P.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(P.FuncFormatter(log_10_product_x))
    ax.yaxis.set_major_formatter(P.FuncFormatter(log_10_product_y))
    ax.plot(x_t, y_t, 'o-', c='r')
    ax.plot(x_r, y_r, 'x-', c='b')
    ax.errorbar(x_r, y_r, yerr=sd, fmt='bx')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    leg = ax.legend(('Targeted removal','Random removal'),loc='lower left')
    for t in leg.get_texts():
        t.set_fontsize('small')
    P.title('Robustness %s'%model)
    P.xlabel('Percentage nodes removed')
    P.ylabel('Relative size of Giant Component')
    P.savefig('../tmp/csen_log_%s.png'%model.replace(' ','_'))
    P.savefig('../tmp/csen_log_%s.eps'%model.replace(' ','_'))
    P.close()
    
###
### Models of network structure
###

##
## Pure structural cohesion example: 2d grid
##
def get_structural_cohesion(n=25):
    return nx.grid_2d_graph(int(n**0.5), int(n**0.5))

##
## Small world model
##
def get_small_world(n=25):
    G = nx.complete_graph(int(n/5))
    nodes = G.nodes()
    for node in nodes:
        new_node = G.order() + 1
        g = nx.complete_graph(4)
        G = nx.disjoint_union(G, g)
        G.add_edge(node, new_node)
    return G

##
## Cohesive small world
##
def get_cohesive_small_world_triangles(n=25, threshold=1.5):
    def make_triangle(G):
        node = random.choice(G.nodes())
        nei = G[node].keys()
        if len(nei) == 2 and not G.has_edge(nei[0],nei[1]):
            G.add_edge(nei[0],nei[1])
        else:
            u = random.choice(nei)
            done = False
            for v in nei:
                if done: break
                if v != u and not G.has_edge(u,v):
                    G.add_edge(u,v)
                    done = True
        return G
    def random_edge(G):
        nodes = G.nodes()
        u = random.choice(nodes)
        v = random.choice(nodes)
        while u == v or G.has_edge(u,v):
            v = random.choice(nodes)
        G.add_edge(u,v)
        return G
    nedges = nx.grid_2d_graph(int(n**0.5),int(n**0.5)).size()
    swi = 0
    runs = 0
    while swi < threshold:
        G = nx.cycle_graph(n)
        while G.size() < nedges:
            for i in range(4):
                G = random_edge(G)
            G = make_triangle(G)
        swi = get_swi(G)
        runs += 1
    msg = "We needed %d runs to obtain a cohesive small world with %d nodes and swi=%.2f"
    print(msg % (runs,n,swi))
    return G


def get_cohesive_small_world(n=25, threshold=1.5):
    def build_net(G,edges):
        nodes = list(G.nodes())
        while G.size() < edges:
            a = random.choice(nodes)
            b = random.choice(nodes)
            if a != b:
                G.add_edge(a,b)
        return G
    nedges = nx.grid_2d_graph(int(n**0.5),int(n**0.5)).size()
    i = 0
    swi = 0
    while swi < threshold or swi > threshold + 0.1:
        seed = nx.cycle_graph(n)
        G = build_net(seed,nedges)
        swi = get_swi(G)
        i += 1
    msg = "We needed %d runs to obtain a cohesive small world with %d nodes and swi=%.1f"
    print(msg % (i,n,threshold))
    return G

##
## Main
##

def main():
    # n has to be a square number (an integer that is the square of an integer)
    # and multiple of 5
    orders = [25, 100]
    nsizes = [150, 75]
    for n in orders:
        print("Generating models with %d nodes" % n)
        # Structural cohesion example
        G_sc = get_structural_cohesion(n=n)
        # Plot it
        draw_graph(G_sc, "structural_cohesion_%d" % n, "Pure Structural Cohesion",
                   node_size=nsizes[orders.index(n)])
        # Component sensitivity
        plot_csen_log(G_sc, "Pure Structural Cohesion %d" % n, n=n)
        print("structural cohesion done!")
        # Small world example
        G_sw = get_small_world(n=n)
        # Plot it
        draw_graph(G_sw, "small_world_%d" % n, "Pure Small World",
                   node_size=nsizes[orders.index(n)])
        # Component sensitivity
        plot_csen_log(G_sw, "Pure Small World %d" % n, n=n)
        print("small world done!")
        # Cohesive small world example
        G_csw = get_cohesive_small_world(n=n)
        # Plot it
        draw_graph(G_csw, "cohesive_small_world_%d" % n, "Cohesive Small World",
                   node_size=nsizes[orders.index(n)])
        # Component sensitivity
        plot_csen_log(G_csw, "Cohesive Small World %d" % n, n=n)
        print("cohesive small world done!")

if __name__ == '__main__':
    main()
