import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union
from itertools import combinations, product
from scipy import stats
import networkx as nx

def causal_graphs_and_structural_equation_models(
    data: pd.DataFrame,
    graph_edges: List[Tuple[str, str]],
    treatment: str,
    outcome: str,
    structural_equations: Optional[Dict[str, str]] = None,
    confounders: Optional[List[str]] = None,
    mediators: Optional[List[str]] = None,
    instruments: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Union[float, List[str], Dict, bool]]:
    """
    Implement Causal Graphs and Structural Equation Models for causal inference.
    
    This function constructs directed acyclic graphs (DAGs), applies d-separation criteria,
    implements backdoor and frontdoor criteria, and estimates causal effects using
    Pearl's causal framework including do-calculus and structural causal models.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing all variables in the causal graph
    graph_edges : List[Tuple[str, str]]
        List of directed edges (parent, child) defining the causal DAG
    treatment : str
        Name of the treatment/intervention variable
    outcome : str
        Name of the outcome variable
    structural_equations : Optional[Dict[str, str]]
        Dictionary mapping variables to their structural equations
    confounders : Optional[List[str]]
        List of potential confounding variables
    mediators : Optional[List[str]]
        List of potential mediating variables
    instruments : Optional[List[str]]
        List of potential instrumental variables
    alpha : float
        Significance level for statistical tests
        
    Returns:
    --------
    Dict containing:
        - 'causal_effect': Estimated average causal effect
        - 'backdoor_valid': Whether backdoor criterion is satisfied
        - 'backdoor_sets': Valid backdoor adjustment sets
        - 'frontdoor_valid': Whether frontdoor criterion is satisfied
        - 'frontdoor_sets': Valid frontdoor adjustment sets
        - 'conditional_independencies': List of conditional independence relationships
        - 'confounders_identified': List of identified confounders
        - 'total_effect': Total causal effect estimate
        - 'direct_effect': Direct causal effect estimate
        - 'indirect_effect': Indirect causal effect estimate
        - 'identification_strategy': Recommended identification strategy
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if not isinstance(graph_edges, list):
        raise ValueError("graph_edges must be a list of tuples")
    if treatment not in data.columns:
        raise ValueError(f"Treatment variable '{treatment}' not found in data")
    if outcome not in data.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in data")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    # Create directed graph using NetworkX
    G = nx.DiGraph()
    all_vars = set()
    for parent, child in graph_edges:
        G.add_edge(parent, child)
        all_vars.update([parent, child])
    
    # Ensure treatment and outcome are in the graph
    if treatment not in all_vars:
        G.add_node(treatment)
        all_vars.add(treatment)
    if outcome not in all_vars:
        G.add_node(outcome)
        all_vars.add(outcome)
    
    # Check if graph is a DAG
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The provided graph contains cycles and is not a DAG")
    
    def d_separated(graph: nx.DiGraph, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z in the DAG.
        Implements d-separation algorithm based on Pearl's criteria.
        """
        # Convert to undirected moral graph for d-separation test
        moral_graph = graph.to_undirected()
        
        # Add edges between parents of common children (moralization)
        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            for p1, p2 in combinations(parents, 2):
                moral_graph.add_edge(p1, p2)
        
        # Remove conditioning set Z
        moral_graph.remove_nodes_from(Z)
        
        # Check if any path exists between X and Y
        for x in X:
            for y in Y:
                if x in moral_graph and y in moral_graph:
                    if nx.has_path(moral_graph, x, y):
                        return False
        return True
    
    def find_backdoor_sets(graph: nx.DiGraph, treatment: str, outcome: str) -> List[Set[str]]:
        """
        Find all valid backdoor adjustment sets.
        A set Z satisfies the backdoor criterion if:
        1. No node in Z is a descendant of treatment
        2. Z blocks every path between treatment and outcome that contains an arrow into treatment
        """
        valid_sets = []
        
        # Get all possible subsets of non-descendants of treatment
        descendants = set(nx.descendants(graph, treatment))
        descendants.add(treatment)  # Treatment itself
        descendants.add(outcome)   # Outcome itself
        
        candidates = set(graph.nodes()) - descendants
        
        # Test all possible subsets
        for r in range(len(candidates) + 1):
            for subset in combinations(candidates, r):
                Z = set(subset)
                
                # Create graph with treatment removed
                G_no_treatment = graph.copy()
                G_no_treatment.remove_node(treatment)
                
                # Check if Z d-separates treatment from outcome in original graph
                # after removing arrows into treatment
                G_backdoor = graph.copy()
                # Remove incoming edges to treatment
                incoming_edges = list(G_backdoor.in_edges(treatment))
                G_backdoor.remove_edges_from(incoming_edges)
                
                if d_separated(G_backdoor, {treatment}, {outcome}, Z):
                    valid_sets.append(Z)
        
        return valid_sets
    
    def find_frontdoor_sets(graph: nx.DiGraph, treatment: str, outcome: str) -> List[Set[str]]:
        """
        Find all valid frontdoor adjustment sets.
        A set Z satisfies the frontdoor criterion if:
        1. Z intercepts all directed paths from treatment to outcome
        2. There is no backdoor path from treatment to Z
        3. All backdoor paths from Z to outcome are blocked by treatment
        """
        valid_sets = []
        
        # Get all nodes between treatment and outcome
        treatment_descendants = set(nx.descendants(graph, treatment))
        outcome_ancestors = set(nx.ancestors(graph, outcome))
        
        candidates = treatment_descendants.intersection(outcome_ancestors)
        
        # Test subsets of candidates
        for r in range(1, len(candidates) + 1):
            for subset in combinations(candidates, r):
                Z = set(subset)
                
                # Check frontdoor criteria
                # Criterion 1: Z intercepts all directed paths from treatment to outcome
                G_no_Z = graph.copy()
                G_no_Z.remove_nodes_from(Z)
                
                paths_blocked = True
                try:
                    if nx.has_path(G_no_Z, treatment, outcome):
                        paths_blocked = False
                except nx.NetworkXNoPath:
                    pass
                
                if not paths_blocked:
                    continue
                
                # Criterion 2: No backdoor path from treatment to Z
                backdoor_free = True
                for z in Z:
                    if not d_separated(graph, {treatment}, {z}, set()):
                        # Check if there's a backdoor path
                        G_undirected = graph.to_undirected()
                        if nx.has_path(G_undirected, treatment, z):
                            # Check if it's not just the direct path
                            G_no_direct = graph.copy()
                            if G_no_direct.has_edge(treatment, z):
                                G_no_direct.remove_edge(treatment, z)
                                if nx.has_path(G_no_direct.to_undirected(), treatment, z):
                                    backdoor_free = False
                                    break
                
                if not backdoor_free:
                    continue
                
                # Criterion 3: All backdoor paths from Z to outcome blocked by treatment
                blocked_by_treatment = True
                for z in Z:
                    if not d_separated(graph, {z}, {outcome}, {treatment}):
                        blocked_by_treatment = False
                        break
                
                if blocked_by_treatment:
                    valid_sets.append(Z)
        
        return valid_sets
    
    def estimate_causal_effect_backdoor(data: pd.DataFrame, treatment: str, 
                                      outcome: str, adjustment_set: Set[str]) -> float:
        """
        Estimate causal effect using backdoor adjustment.
        E[Y|do(X=1)] - E[Y|do(X=0)] = Σ_z [E[Y|X=1,Z=z] - E[Y|X=0,Z=z]] * P(Z=z)
        """
        if not adjustment_set:
            # Simple difference in means
            treated = data[data[treatment] == 1][outcome].mean()
            control = data[data[treatment] == 0][outcome].mean()
            return treated - control
        
        # Stratified adjustment
        total_effect = 0
        adjustment_vars = list(adjustment_set)
        
        # Group by adjustment variables
        grouped = data.groupby(adjustment_vars)
        
        for group_vals, group_data in grouped:
            if len(group_data) < 2:
                continue
                
            # Calculate conditional expectation for this stratum
            treated_group = group_data[group_data[treatment] == 1]
            control_group = group_data[group_data[treatment] == 0]
            
            if len(treated_group) > 0 and len(control_group) > 0:
                treated_mean = treated_group[outcome].mean()
                control_mean = control_group[outcome].mean()
                stratum_effect = treated_mean - control_mean
                
                # Weight by probability of this stratum
                stratum_prob = len(group_data) / len(data)
                total_effect += stratum_effect * stratum_prob
        
        return total_effect
    
    def estimate_causal_effect_frontdoor(data: pd.DataFrame, treatment: str,
                                       outcome: str, mediator_set: Set[str]) -> float:
        """
        Estimate causal effect using frontdoor adjustment.
        More complex calculation involving mediating variables.
        """
        # Simplified frontdoor estimation
        # This would require more sophisticated implementation for multiple mediators
        if len(mediator_set) == 1:
            mediator = list(mediator_set)[0]
            
            # Step 1: Effect of treatment on mediator
            treated_mediator = data[data[treatment] == 1][mediator].mean()
            control_mediator = data[data[treatment] == 0][mediator].mean()
            
            # Step 2: Effect of mediator on outcome (controlling for treatment)
            # This is a simplified version
            from sklearn.linear_model import LinearRegression
            
            X = data[[treatment, mediator]]
            y = data[outcome]
            
            model = LinearRegression().fit(X, y)
            mediator_coef = model.coef_[1]  # Coefficient for mediator
            
            # Front