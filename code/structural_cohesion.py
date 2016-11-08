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
from project import results_dir, tables_dir, python_years, debian_years
from project import python_releases, debian_releases
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
