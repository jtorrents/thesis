# Paths for the project
import os
import utils

# Directories
# The root directory is the folder in which this file lives
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, os.pardir, 'data')
plots_dir = os.path.join(root, os.pardir, 'figures')
results_dir = os.path.join(root, os.pardir, 'results')
tmp_dir = os.path.join(root, os.pardir, 'tmp')
tables_dir = os.path.join(root, os.pardir, 'tables')

# Defaults
default_years = list(range(1991, 2015))
python_years = list(range(1999, 2015))
default_branches = ['2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7',
                    '3.0', '3.1', '3.2', '3.3', '3.4']
                    #, 'default', 'legacy-trunk']
python_releases = default_branches
# Debian years and releases
debian_years = list(range(1999, 2013))
debian_releases = ['Slink', 'Potato', 'Woody', 'Sarge', 'Etch', 'Lenny', 'Squeeze']
# Bipartite Multigraph graphml file with all python networks
multigraph_file = os.path.join(data_dir, "cpython_multigraph.graphml.gz")

# Dataframes for regression modeling
collaboration_file = os.path.join(data_dir, "developer_contributions_df.csv")
survival_file = os.path.join(data_dir, "survival_python_df.csv")
lifetime_file = os.path.join(data_dir, "lifetime_python_df.csv")

# Network varaibles
connectivity_file = os.path.join(results_dir, "k_components_years_201504242234.pkl")
connectivity_file_branches = os.path.join(results_dir, "k_components_branches_201504242308.pkl")
centrality_file = os.path.join(results_dir, "bipartite_centrality_years_201504242313.pkl")
centrality_file_branches = os.path.join(results_dir, "bipartite_centrality_branches_201504242323.pkl")
layouts_file = os.path.join(results_dir, "layouts_years_201504242325.pkl")
layouts_file_branches = os.path.join(results_dir, "layouts_branches_201504230813.pkl")

##
## Python analysis results
##
# Connectivity
python_connectivity_years_file = os.path.join(results_dir, 'structural_cohesion_python_201611061643.pkl')
python_connectivity_releases_file = os.path.join(results_dir, 'structural_cohesion_python_releases_201611062044.pkl')
python_connectivity_null_model_years_file = os.path.join(results_dir, 'structural_cohesion_null_model_python_years_201611081322.pkl')
python_connectivity_null_model_releases_file = os.path.join(results_dir, 'structural_cohesion_null_model_python_releases_201611081338.pkl')
# Small world metrics
python_small_world_years_file = os.path.join(results_dir, 'small_world_python_years_201611071443.pkl')
python_small_world_releases_file = os.path.join(results_dir, 'small_world_python_releases_201611071502.pkl')
# Network Layouts
python_layouts_years_file = os.path.join(results_dir, 'layouts_python_201611061643.pkl')
python_layouts_releases_file = os.path.join(results_dir, 'layouts_python_releases_201611062153.pkl')

##
## Debian analysis results
##
# Connectivity
debian_connectivity_years_file = os.path.join(results_dir, 'structural_cohesion_debian_201611061716.pkl')
debian_connectivity_releases_file = os.path.join(results_dir, 'structural_cohesion_debian_releases_201611062047.pkl')
debian_connectivity_null_model_years_file = os.path.join(results_dir, 'structural_cohesion_null_model_debian_years_201611082001.pkl')
debian_connectivity_null_model_releases_file = os.path.join(results_dir, 'structural_cohesion_null_model_debian_releases_201611082023.pkl')
# Small world metrics
debian_small_world_years_file = os.path.join(results_dir, 'small_world_debian_years_201611141323.pkl')
debian_small_world_releases_file = os.path.join(results_dir, 'small_world_debian_201611052103.pkl')
# Network Layouts
debian_layouts_years_file = os.path.join(results_dir, 'layouts_debian_201611061716.pkl')
debian_layouts_releases_file = os.path.join(results_dir, 'layouts_debian_releases_201611062157.pkl')

##
## Functions to load results
##
def get_structural_cohesion_results(project_name, kind):
    if project_name == 'python':
        if kind == 'years':
            result = utils.load_result_pkl(python_connectivity_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(python_connectivity_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    elif project_name == 'debian':
        if kind == 'years':
            result = utils.load_result_pkl(debian_connectivity_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(debian_connectivity_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    else:
        raise Exception('Unknown project name {}'.format(project_name))
    return result


def get_structural_cohesion_null_model_results(project_name, kind):
    if project_name == 'python':
        if kind == 'years':
            result = utils.load_result_pkl(python_connectivity_null_model_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(python_connectivity_null_model_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    elif project_name == 'debian':
        if kind == 'years':
            result = utils.load_result_pkl(debian_connectivity_null_model_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(debian_connectivity_null_model_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    else:
        raise Exception('Unknown project name {}'.format(project_name))
    return result


def get_small_world_results(project_name, kind):
    if project_name == 'python':
        if kind == 'years':
            result = utils.load_result_pkl(python_small_world_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(python_small_world_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    elif project_name == 'debian':
        if kind == 'years':
            result = utils.load_result_pkl(debian_small_world_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(debian_small_world_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    else:
        raise Exception('Unknown project name {}'.format(project_name))
    return result


def get_layouts(project_name, kind):
    if project_name == 'python':
        if kind == 'years':
            result = utils.load_result_pkl(python_layouts_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(python_layouts_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    elif project_name == 'debian':
        if kind == 'years':
            result = utils.load_result_pkl(debian_layouts_years_file)
        elif kind == 'releases':
            result = utils.load_result_pkl(debian_layouts_releases_file)
        else:
            raise Exception('Unknown kind {}'.format(kind))
    else:
        raise Exception('Unknown project name {}'.format(project_name))
    return result
