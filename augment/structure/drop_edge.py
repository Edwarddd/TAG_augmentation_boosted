import numpy as np
import torch

from utils import get_degrees


def drop_edge_degree(graph, augment_ratio):
    # Get the number of nodes and edges
    node_num, _ = graph.x.size()
    edge_index = graph.edge_index
    edge_num = edge_index.size(1)
    
    # Calculate the number of edges to drop and select
    drop_num = int(edge_num * augment_ratio)
    select_num = edge_num - drop_num

    # Get degrees of nodes
    degree = get_degrees(graph)

    # Sort the degrees and get the indices
    sorted_degree, indices = torch.sort(degree)
    
    # Convert edge_index to a list of tuples for easier manipulation
    edge_index_np = edge_index.cpu().numpy().T
    edge_index_list = list(map(tuple, edge_index_np))

    # Determine edges to drop based on node degrees
    selected_edges = []
    drop_set = set()

    # Iterate over nodes with sorted degrees
    for node in indices:
        if len(drop_set) >= drop_num:
            break
        # Iterate over edges
        for edge in edge_index_list:
            if edge[0] == node.item() or edge[1] == node.item():
                if len(drop_set) < drop_num:
                    drop_set.add(edge)
                else:
                    break

    # Select the edges that are not in the drop set
    selected_edges = [edge for edge in edge_index_list if edge not in drop_set]
    selected_edges_np = np.array(selected_edges).T

    # Update the graph's edge_index with the selected edges
    graph.edge_index = torch.tensor(selected_edges_np, dtype=torch.long)

    return graph
