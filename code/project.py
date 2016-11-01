# Paths for the project
import os

# Directories
# The root directory is the folder in which this file lives
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, os.pardir, 'data')
plots_dir = os.path.join(root, os.pardir, 'figures')
results_dir = os.path.join(root, os.pardir, 'results')
tmp_dir = os.path.join(root, os.pardir, 'tmp')

# Bipartite Multigraph graphml file
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

# Defaults
default_years = list(range(1991, 2015))
default_branches = ['2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7',
                    '3.0', '3.1', '3.2', '3.3', '3.4']
                    #, 'default', 'legacy-trunk']
