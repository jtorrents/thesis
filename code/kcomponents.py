""" Fast approximation for k-component structure
"""
#    Copyright (C) 2015 by 
#    Jordi Torrents <jtorrents@milnou.net>
#    All rights reserved.
#    BSD license.
import itertools
import collections

import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.algorithms.connectivity import cuts
from networkx.algorithms.connectivity import kcomponents as ex
from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
from networkx.algorithms.connectivity import local_node_connectivity as exact_local_node_connectivity
from networkx.algorithms.flow import build_residual_network

from antigraph import AntiGraph

__author__ = """\n""".join(['Jordi Torrents <jtorrents@milnou.net>'])

__all__ = [
    'k_components',
    'k_components_approx_accuracy',
]

def k_components(G, average=True, exact=False, min_density=0.95):
    r"""Returns the approximate k-component structure of a graph G.
    
    A `k`-component is a maximal subgraph of a graph G that has, at least, 
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected 
    graph can contain several 2-components, each of which can contain 
    one or more 3-components, and so forth.

    This implementation is based on the fast heuristics to approximate
    the `k`-component sturcture of a graph [1]_. Which, in turn, it is based on
    a fast approximation algorithm for finding good lower bounds of the number 
    of node independent paths between two nodes [2]_.
  
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    average : Boolean (default=True)
        Compute also the average connectivity of each `k`-component

    exact : Bool
        If True use the exact flow based connectivity to compute pairwise
        node connectivity. If False use White and Newman's fast 
        approximation. Default value: False.

    min_density : Float
        Density relaxation treshold. Default value 0.95

    Returns
    -------
    k_components : dict
        Dictionary with connectivity level `k` as key and a list of
        sets of nodes that form a k-component of level `k` as values.
 
    k_number : dict
        Dictionary with nodes as keys with value of the maximum k of the 
        deepest k-component in which they are embedded.


    Examples
    --------
    >>> # Petersen graph has 10 nodes and it is triconnected, thus all 
    >>> # nodes are in a single component on all three connectivity levels
    >>> from networkx.algorithms import approximation as apxa
    >>> G = nx.petersen_graph()
    >>> k_components = apxa.k_components(G)
    
    Notes
    -----
    The logic of the approximation algorithm for computing the `k`-component 
    structure [1]_ is based on repeatedly applying simple and fast algorithms 
    for `k`-cores and biconnected components in order to narrow down the 
    number of pairs of nodes over which we have to compute White and Newman's
    approximation algorithm for finding node independent paths [2]_. More
    formally, this algorithm is based on Whitney's theorem, which states 
    an inclusion relation among node connectivity, edge connectivity, and 
    minimum degree for any graph G. This theorem implies that every 
    `k`-component is nested inside a `k`-edge-component, which in turn, 
    is contained in a `k`-core. Thus, this algorithm computes node independent
    paths among pairs of nodes in each biconnected part of each `k`-core,
    and repeats this procedure for each `k` from 3 to the maximal core number 
    of a node in the input graph.

    Because, in practice, many nodes of the core of level `k` inside a 
    bicomponent actually are part of a component of level k, the auxiliary 
    graph needed for the algorithm is likely to be very dense. Thus, we use 
    a complement graph data structure (see `AntiGraph`) to save memory. 
    AntiGraph only stores information of the edges that are *not* present 
    in the actual auxiliary graph. When applying algorithms to this 
    complement graph data structure, it behaves as if it were the dense 
    version.

    See also
    --------
    k_components

    References
    ----------
    .. [1]  Torrents, J. and F. Ferraro (2015) Structural Cohesion: 
            Visualization and Heuristics for Fast Computation.
            http://arxiv.org/pdf/1503.04476v1

    .. [2]  White, Douglas R., and Mark Newman (2001) A Fast Algorithm for 
            Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
            http://eclectic.ss.uci.edu/~drwhite/working.pdf

    .. [3]  Moody, J. and D. White (2003). Social cohesion and embeddedness: 
            A hierarchical conception of social groups. 
            American Sociological Review 68(1), 103--28.
            http://www2.asanet.org/journals/ASRFeb03MoodyWhite.pdf

    """
    # Dictionary with connectivity level (k) as keys and a list of
    # sets of nodes that form a k-component as values
    k_components = collections.defaultdict(list)
    # Dictionary with nodes as keys and maximum k of the deepest 
    # k-component in which they are embedded
    k_number = dict.fromkeys(G, 0)
    # dict to store node independent paths
    nip = {} 
    def _update_results(k, components, avg_k=None):
        average = True if avg_k is not None else False
        for component in components:
            if len(component) > k:
                if average:
                    k_components[k].append((avg_k, set(component)))
                else:
                    k_components[k].append(set(component))
                for node in component:
                    if average:
                        k_number[node] = (k, avg_k)
                    else:
                        k_number[node] = k
    if exact:
        node_connectivity = exact_local_node_connectivity
        min_density = 1.0
        A = build_auxiliary_node_connectivity(G)
        R = build_residual_network(A, 'capacity')
    else:
        node_connectivity = local_node_connectivity
    # make a few functions local for speed
    k_core = nx.k_core
    core_number = nx.core_number
    biconnected_components = nx.biconnected_components
    density = nx.density
    combinations = itertools.combinations
    # Exact solution for k = {1,2}
    # There is a linear time algorithm for triconnectivity, if we had an
    # implementation available we could start from k = 4.
    if average:
        _update_results(1, nx.connected_components(G), 1)
        _update_results(2, biconnected_components(G), 2)
    else:
        _update_results(1, nx.connected_components(G))
        _update_results(2, biconnected_components(G))
    # There is no k-component of k > maximum core number
    # \kappa(G) <= \lambda(G) <= \delta(G)
    g_cnumber = core_number(G)
    max_core = max(g_cnumber.values())
    for k in range(3, max_core + 1):
        C = k_core(G, k, core_number=g_cnumber)
        for nodes in biconnected_components(C):
            # Build a subgraph SG induced by the nodes that are part of
            # each biconnected component of the k-core subgraph C.
            if len(nodes) < k:
                continue
            SG = G.subgraph(nodes)
            if exact:
                ar_nodes = [n for n, d in A.nodes(data=True) if d['id'] in nodes]
                SA = A.subgraph(ar_nodes)
                SR = R.subgraph(ar_nodes)
                kwargs = dict(auxiliary=SA, residual=SR)
            else:
                kwargs = dict()
            # Build auxiliary graph
            H = AntiGraph()
            H.add_nodes_from(SG.nodes())
            for u,v in combinations(SG, 2):
                if exact:
                    kwargs['cutoff'] = k
                K = node_connectivity(SG, u, v, **kwargs)
                nip[(u, v)] = K
                if k > K:
                    H.add_edge(u,v)
            for h_nodes in biconnected_components(H):
                if len(h_nodes) <= k:
                    continue
                SH = H.subgraph(h_nodes)
                for Gc in _cliques_heuristic(SG, SH, k, min_density):
                    for k_nodes in biconnected_components(Gc):
                        Gk = nx.k_core(SG.subgraph(k_nodes), k)
                        if len(Gk) <= k:
                            continue
                        #k_components[k].append(set(Gk))
                        if average:
                            num = 0.0
                            den = 0.0
                            for u, v in combinations(Gk, 2):
                                den += 1
                                num += nip.get((u, v), nip.get((v, u)))
                            _update_results(k, [list(Gk.nodes())], (num/den))
                        else:
                            _update_results(k, [Gk.nodes()])
    return k_components, k_number


def _cliques_heuristic(G, H, k, min_density):
    h_cnumber = nx.core_number(H)
    for i, c_value in enumerate(sorted(set(h_cnumber.values()), reverse=True)):
        cands = set(n for n, c in h_cnumber.items() if c == c_value)
        # Skip checking for overlap for the highest core value
        if i == 0:
            overlap = False
        else:
            overlap = set.intersection(*[
                        set(x for x in H[n] if x not in cands)
                        for n in cands])
        if overlap and len(overlap) < k:
            SH = H.subgraph(cands | overlap)
        else:
            SH = H.subgraph(cands)
        sh_cnumber = nx.core_number(SH)
        SG = nx.k_core(G.subgraph(SH), k)
        while not (_same(sh_cnumber) and nx.density(SH) >= min_density):
            SH = H.subgraph(SG)
            if len(SH) <= k:
                break
            sh_cnumber = nx.core_number(SH)
            sh_deg = dict(SH.degree())
            min_deg = min(sh_deg.values())
            SH.remove_nodes_from(n for n, d in sh_deg.items() if d == min_deg)
            SG = nx.k_core(G.subgraph(SH), k)
        else:
            yield SG


def _same(measure, tol=0):
    vals = set(measure.values())
    if (max(vals) - min(vals)) <= tol:
        return True
    return False


def build_auxiliary_graph(G, k, exact=False, antigraph=False):
    core_number = nx.core_number(G)
    C = G.subgraph((n for n in G if core_number[n] >= k))
    if antigraph:
        H = AntiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(C.nodes())
    for u,v in itertools.combinations(C, 2):
        if exact:
            K = nx.local_node_connectivity(C, u, v)
        else:
            K = local_node_connectivity(C, u, v, max_paths=k)
        if antigraph:
            if k > K:
                H.add_edge(u,v)
        else:
            if k <= K:
                H.add_edge(u,v)
    return H


def k_components_approx_accuracy(G, k_components, 
                                    min_order=3, verbose=True, cexact=False):
    r"""Test accuracy of the `k`-component approximation algorithm. 

    For each detetcted `k`-component with the approximation algorithm, this
    function tests that it actually has node connectivity `k`. If not, the
    exact algorithm is used to compute the actual `k`-component structure.
   
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    k_components : dict
        Dictionary with the k-component structure of G as returned by
        approximation.k_components

    min_order : integer (default=3)
        `k`-components with order below `min_order` are not considered

    verbose : boolean (default=True)
        Print to the standard output information on the progress of the
        computation

    Returns
    -------
    result : dict
        A dictionary with connectivity levels as keys, and another dictionary
        for each `k`-component as values with information abouts its accuracy

    See also
    --------
    node_connectivity
    approximation.k_components
    k_components

    References
    ----------
    .. [1]  Torrents, J. and F. Ferraro. Structural Cohesion: a theoretical 
            extension and a fast approximation algorithm. Draft
            http://www.milnou.net/~jtorrents/structural_cohesion.pdf

    """
    result = collections.defaultdict(list)
    for approx_k in range(1, max(k_components.keys())+1):
        if approx_k not in k_components or approx_k < 3:
            continue
        for avg_k, candidates in k_components[approx_k]:
            if not candidates:
                continue
            k_component = G.subgraph(candidates)
            if k_component.order() < min_order:
                continue
            cut_set = cuts.minimum_node_cut(k_component)
            actual_k = len(cut_set)
            correct = actual_k >= approx_k
            if not correct:
                if verbose:
                    msg = "Error in k=%s with %d candidates."
                    print(msg % (approx_k,len(candidates)))
                if cexact:
                    exact_kc, exact_knum = ex.k_components(k_component)
                    if approx_k in exact_kc:
                        exact = [len(c) for c in exact_kc[approx_k]]
                    else:
                        exact = "Doesn't exists"
                else:
                    exact = "Not computed"
            else:
                exact = None

            result[approx_k].append({   'n': k_component.order(),
                                        'avg_k': avg_k,
                                        'approx_k': approx_k,
                                        'actual_k': actual_k,
                                        'correct': correct,
                                        'cut_set': cut_set,
                                        'exact': exact})
            if verbose:
                msg = '%s: %d nodes in %d-component with connectivity %d (%.3f)'
                m = "Correct" if correct else "Error"
                print(msg % (m, k_component.order(), approx_k, actual_k, avg_k))
    return dict(result)
