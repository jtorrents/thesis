import argparse
from datetime import datetime
import sys
import os

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from kcomponents import k_components
from networks import debian_networks_by_year, python_networks_by_year
from networks import debian_networks_by_release, python_networks_by_branch
from null_model import generate_random_configuration_2mode
from project import results_dir, tables_dir, tmp_dir
from project import python_years, debian_years, python_releases, debian_releases
from project import get_structural_cohesion_results, get_layouts
from project import get_structural_cohesion_null_model_results
import utils


now = datetime.now().strftime('%Y%m%d%H%M')

##
## Heuristics for computing k-components
##
def compute_k_components(names_and_networks, project_name, kind, now=now):
    result = {}
    print("Computing k-components for {} networks".format(project_name))
    for name, G in names_and_networks():
        result[name] = {}
        print("    Analizing {} network".format(name))
        k_comp, k_num = k_components(G)
        result[name]['k_components'] = k_comp
        result[name]['k_num'] = k_num
    fname="{0}/structural_cohesion_{1}_{2}_{3}.pkl".format(results_dir,
                                                           project_name,
                                                           kind, now)
    utils.write_results_pkl(result, fname)


##
## Compute k-componets for null models
##
def compute_k_components_null_model(names_and_networks, project_name, kind, now=now):
    result = {}
    print("Computing k-components for {} null models".format(project_name))
    for name, G in names_and_networks():
        result[name] = {}
        print("    Analizing null model for {} network".format(name))
        G_random = generate_random_configuration_2mode(G)
        k_comp, k_num = k_components(G_random)
        result[name]['k_components'] = k_comp
        result[name]['k_num'] = k_num
    fname="{0}/structural_cohesion_null_model_{1}_{2}_{3}.pkl".format(results_dir,
                                                                      project_name,
                                                                      kind, now)
    utils.write_results_pkl(result, fname)



##
## Compute layouts for 3D scatter plots and network drawings
##
def compute_layouts(names_and_networks, project_name, kind, now=now):
    result = {}
    print("Computing layouts for {} networks".format(project_name))
    for name, G in names_and_networks():
        print("    {} network".format(name))
        result[name] = graphviz_layout(G, prog='neato')
    fname = "{0}/layouts_{1}_{2}_{3}.pkl".format(results_dir, project_name,
                                                 kind, now)
    utils.write_results_pkl(result, fname)


##
## Write Structural Cohesion analysis table (actual and null model)
##
def write_structural_cohesion_table(actual, random, project_name, kind):
    fname = 'table_structural_cohesion_{}_{}.tex'.format(project_name, kind)
    first = 'Years' if kind == 'years' else 'Names'
    if project_name == 'python':
        order = python_years if kind == 'years' else python_releases
    if project_name == 'debian':
        order = debian_years if kind == 'years' else debian_releases
    with open(os.path.join(tables_dir, fname), 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\begin{center}\n")
        #f.write("\\begin{footnotesize}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
        f.write("\hline\n")
        header = '%s&Nodes&GC&Random GC&GBC&Random GBC&maximum $k$&Random max $k$\\\\\n'
        f.write(header % first)
        f.write("\hline\n")
        for name in sorted(actual, key=order.index):
            act = actual[name]['k_components']
            max_act = max(act.keys())
            rand = random[name]['k_components']
            max_rand = max(rand.keys())
            total = sum(len(component) for k, component in act[1])
            row = '{:s}&{:,}&{:.1f}\%&{:.1f}\%&{:.1f}\%&{:.1f}\%&{:d} ({:.1f}\%)&{:d} ({:.1f}\%)\\\\\n'
            f.write(row.format(
                        str(name),
                        total,
                        (max(len(c) for k, c in act[1]) / total) * 100,
                        (max(len(c) for k, c in rand[1]) / total) * 100,
                        (max(len(c) for k, c in act[2]) / total) * 100,
                        (max(len(c) for k, c in rand[2]) / total) * 100,
                        max_act,
                        (max(len(c) for k, c in act[max_act]) / total) * 100,
                        max_rand,
                        (max(len(c) for k, c in rand[max_rand]) / total) * 100,
                        ))
        f.write("\hline\n")
        f.write("\end{tabular}\n")
        #f.write("\end{footnotesize}\n")
        f.write("\caption{Structural Cohesion metrics for %s networks.}\n" % project_name)
        f.write("\label{str_cohesion_%s}\n" % project_name)
        f.write("\end{center}\n")
        f.write("\end{table}\n")
        f.write("\n")


##
## 3d scatter plots for average connectivity
##
def scatter_3d_connectivity(G, k_number, pos, name, project_name,
                            max_k=None, null=False, directory=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    if project_name == 'python':
        top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite']==1}
        size_top, size_bot = 150, 100
    else:
        top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
        size_top, size_bot = 65, 35

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')

    x_top = np.asarray([xy[0] for n, xy in pos.items() if n in top_nodes])
    y_top = np.asarray([xy[1] for n, xy in pos.items() if n in top_nodes])
    z_top = np.asarray([k_number[n][1] for n in pos if n in top_nodes])

    x_bot = np.asarray([xy[0] for n, xy in pos.items() if n not in top_nodes])
    y_bot = np.asarray([xy[1] for n, xy in pos.items() if n not in top_nodes])
    z_bot = np.asarray([k_number[n][1] for n in pos if n not in top_nodes])

    # get extrema values.
    xmin, xmax = min(x_top.min(), x_bot.min()), max(x_top.max(), x_bot.max())
    ymin, ymax = min(y_top.min(), y_bot.min()), max(y_top.max(), y_bot.max())
    zmin, zmax = min(z_top.min(), z_bot.min()), max(z_top.max(), z_bot.max())

    color_map = cm.get_cmap('viridis_r', zmax if max_k is None else max_k)
    #color_map = cm.get_cmap('inferno_r', zmax if max_k is None else max_k)
    #color_map = cm.get_cmap('magma_r', zmax if max_k is None else max_k)

    scatter_top = ax.scatter(x_top, y_top, z_top, marker='o', s=size_top,
                                c=z_top, vmin=zmin, cmap=color_map,
                                vmax=zmax if max_k is None else max_k)
    scatter_bot = ax.scatter(x_bot, y_bot, z_bot, marker='^', alpha=0.65, s=size_bot,
                                c=z_bot, vmin=zmin, cmap=color_map,
                                vmax=zmax if max_k is None else max_k)

    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax if max_k is None else max_k)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'Average connectivity $\bar(\kappa)$')
    fig.colorbar(scatter_top, shrink=0.5, aspect=9,
                 ticks=range(1, int(zmax)+1 if max_k is None else int(max_k)+1))

    if null:
        plt.title('Null model {} 3d scatter plot {}'.format(project_name.capitalize(), name))
    else:
        plt.title('{} 3d scatter plot {}'.format(project_name.capitalize(), name))

    if directory is None:
        plt.ion()
        plt.show()
    else:
        # Save plot and close
        if null:
            plt.savefig('{}/3d_scatter_{}_{}_null.pdf'.format(directory, project_name, name))
            plt.savefig('{}/3d_scatter_{}_{}_null.png'.format(directory, project_name, name))
        else:
            plt.savefig('{}/3d_scatter_{}_{}.pdf'.format(directory, project_name, name))
            plt.savefig('{}/3d_scatter_{}_{}.png'.format(directory, project_name, name))
        plt.close()


def all_3d_scatter_plots(names_and_networks, project_name, kind, directory=tmp_dir):
    layouts = get_layouts(project_name, kind)
    conn = get_structural_cohesion_results(project_name, kind)
    conn_null = get_structural_cohesion_null_model_results(project_name, kind)

    print('Scatter 3d plots for {}:'.format(project_name))
    for name, G in names_and_networks():
        k_number = conn[name]['k_num']
        max_k = max(ak for k, ak in k_number.values())
        k_number_null = conn_null[name]['k_num']
        pos = layouts[name]
        print('    Plotting {} actual network'.format(name))
        scatter_3d_connectivity(G, k_number, pos, name, project_name,
                                max_k=None, null=False, directory=directory)
        print('    Plotting {} null model network'.format(name))
        Gr = generate_random_configuration_2mode(G)
        pos_r = utils.relabel_layout(G, Gr, pos)
        scatter_3d_connectivity(Gr, k_number_null, pos_r, name, project_name,
                                max_k=max_k, null=True, directory=directory)


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

    group_type = parser.add_mutually_exclusive_group(required=True)
    group_type.add_argument('-k', '--kcomponents', action='store_true',
                            help='Compute k-components')
    group_type.add_argument('-n', '--null_model', action='store_true',
                            help='Compute k-components for null models')
    group_type.add_argument('-l', '--layouts', action='store_true',
                            help='Compute layouts')
    group_type.add_argument('-f', '--figures', action='store_true',
                            help='Make 3d scatter plots')

    group_kind = parser.add_mutually_exclusive_group(required=True)
    group_kind.add_argument('-y', '--years', action='store_true',
                            help='Use year based networks')
    group_kind.add_argument('-r', '--releases', action='store_true',
                            help='Use release based networks')

    args = parser.parse_args()

    if args.kcomponents:
        function = compute_k_components
    elif args.null_model:
        function = compute_k_components_null_model
    elif args.layouts:
        function = compute_layouts
    elif args.figures:
        function = all_3d_scatter_plots

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
    
    # Run what is asked in the command line
    function(names_and_networks, project_name, kind)
    #print(names_and_networks, project_name, kind)

if __name__ == '__main__':
    main()
