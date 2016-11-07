import argparse
from datetime import datetime
import sys

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from kcomponents import k_components
from networks import debian_networks_by_year, python_networks_by_year
from networks import debian_networks_by_release, python_networks_by_branch
from project import results_dir
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
