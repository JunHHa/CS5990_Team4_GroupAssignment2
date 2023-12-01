import heapq
from itertools import count
from mpi4py import MPI
import networkx as nx
import pickle
import csv
import sys
import time
from datetime import datetime


def progress_bar(percent, complete=False):
    bar_length = 50
    block = int(round(bar_length * percent))
    progress = "=" * block + "-" * (bar_length - block)
    sys.stdout.write(f"\r[{progress}] {int(percent * 100)}%")

    if complete:
        sys.stdout.write("\n")

    sys.stdout.flush()


def distribute_nodes(nodes, rank, size):
    # Split the dictionary keys
    keys = list(nodes)
    total_nodes = len(nodes)
    keys_per_process = total_nodes // size
    remainder = total_nodes % size

    # Calculate the start and end indices for each process
    if rank < remainder:
        # Give one extra node to the first 'remainder' processes
        start = rank * (keys_per_process + 1)
        end = start + keys_per_process + 1
    else:
        # The rest of the processes get keys_per_process nodes
        start = remainder * (keys_per_process + 1) + (rank - remainder) * keys_per_process
        end = start + keys_per_process

    return keys[start:end]


def closeness_centrality(dist, graph):
    centrality = {}

    for node, d in dist.items():
        total_distance = sum(d.values())
        centrality[node] = (len(graph) - 1) / total_distance

    return centrality

def betweenness_centrality(G, weight='weight'):
    nodes = G.nodes()  # Get all the nodes in the graph G

    # Initialize a dictionary to store betweenness centrality for each node
    betweenness = {v: 0 for v in nodes}

    # Iterate over each node in the graph as the source node
    for s in nodes:
        # sigma = Tracks the number of shortest paths from source 's' to each node
        sigma = dict.fromkeys(nodes, 0)
        sigma[s] = 1  # The number of paths from 's' to itself is always 1

        # D = Tracks the shortest distance from 's' to each node
        D = dict.fromkeys(nodes, float('inf'))  # Initialize with infinite distance
        D[s] = 0  # Distance from 's' to itself is 0

        # ancestors = Tracks the predecessors of each node in the path from 's'
        ancestors = {v: [] for v in nodes}

        # Prepare for Dijkstra's algorithm
        seen = {s: 0}
        c = count()
        Q = []  # Priority queue for nodes to visit
        heapq.heappush(Q, (0, next(c), s))  # Start with the source node 's'

        # Execute Dijkstra's algorithm; couldn't get the standard nx.Dijkstra's algorithm to work to give me the correct values. Would give me values > 1
        while Q:
            # Get the node with the smallest distance
            (dist, _, v) = heapq.heappop(Q)

            # Skip if the current distance is not the shortest
            if dist > D[v]:
                continue

            # Explore neighbors of the current node
            for w, edgedata in G[v].items():
                vw_dist = dist + edgedata.get(weight, 1)  # Calculate distance to neighbor
                if vw_dist < D[w]:
                    # Update shortest distance and path for neighbor
                    D[w] = vw_dist
                    heapq.heappush(Q, (vw_dist, next(c), w))
                    sigma[w] = sigma[v]  # Update shortest path count
                    ancestors[w] = [v]  # Set current node as predecessor
                elif vw_dist == D[w]:  # If a new shortest path is found
                    sigma[w] += sigma[v]  # Update the count of shortest paths
                    ancestors[w].append(v)  # Add current node as another predecessor

        # Accumulate betweenness values
        delta = dict.fromkeys(nodes, 0)  # Tracks the dependency of each node
        for v in sorted(D, key=D.get, reverse=True):
            for w in ancestors[v]:
                # Calculate dependency value
                delta[w] += (sigma[w] / sigma[v]) * (1 + delta[v])
            if v != s:
                betweenness[v] += delta[v]  # Add dependency value to betweenness

    # Normalize the betweenness values for undirected graphs
    n = len(nodes)  # Number of nodes in the graph
    if n > 1:
        normal = 1 / ((n - 1) * (n - 2) / 1)  # Calculate normalization; for some reason I get double value if using standard 1 / ((n - 1) * (n - 2) / 2), so I have this
        for v in betweenness:
            betweenness[v] *= normal  # Apply normalization

    return betweenness  # Return the calculated betweenness centrality values


def process_data(graph, nodes, rank):
    dist = {}
    centrality = {}
    betweenness = {}
    networkx_betweenness = {}
    sum_cent = 0
    count = 0
    start_time = time.time()  # Record the start time
    # print(f"start time: ", start_time)
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Current time in 24-hour format: {current_time}")

    # uncomment the following line and code before forloop ends for real time updates for elapsed time
    # interval = 10  # Print elapsed time every 10 seconds

    output_file = 'concatenated_result.txt'
    with open(output_file, 'w') as file:
        for node in nodes:
            # Update the loading bar
            percent_complete = (count + 1) / len(nodes)
            progress_bar(percent_complete)

            dist[node], paths = nx.single_source_dijkstra(graph, node)
            centrality = closeness_centrality(dist, graph)
            file.write(f"Node {node}: Closeness Centrality = {centrality[node]:}\n")  # Write each node's centrality
            count += 1
            
            sum_cent += centrality[node] # finding the sum of closeness for average
            
            # betweenness = betweenness_centrality(graph)                                                       # create betweenness
            # networkx_betweenness = nx.betweenness_centrality(graph)                                           # create networkx betweenness
            # file.write(f"Node {node}: Betweenness Centrality = {betweenness[node]:}\n")                       # make file write for betweenness values
            # file.write(f"Node {node}: NetworkX betweenness Centrality = {networkx_betweenness[node]:}\n")     # make file write for networkx betweenness values
            # count += 1
            
            # sum_cent += betweenness[node]    # finding the sum of betweenness for average

            # Print elapsed time at specified interval
            # if time.time() - start_time > interval:
            #     elapsed_time = time.time() - start_time
            #     hours = int(elapsed_time // 3600)
            #     minutes = int((elapsed_time % 3600) // 60)
            #     seconds = int(elapsed_time % 60)
            #     print(f" Elapsed time: {hours:02}:{minutes:02}:{seconds:02}")
            #
            #     interval += 10  # Increase interval for the next print
            
        file.write(f"Average Closeness Centrality = {sum_cent/(len(nodes))}\n")      # file write the average of closeness
        # file.write(f"Average Betweenness Centrality = {sum_cent/(len(nodes))}\n")  # file write the average of betweenness

    # Finish the loading bar
    progress_bar(1, complete=True)

    # Print final elapsed time
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {hours:02}:{minutes:02}:{seconds:02}")

    
    return centrality


def load_data(filename, print=False):
    with open(filename) as fin, open('fixed' + filename, 'w') as fout:
        for line in fin:
            fout.write(line.replace(' ', ','))
    fin.close()
    fout.close()
    filename = 'fixed' + filename

    with open(filename, 'r') as nodecsv:
        nodereader = csv.reader(nodecsv)
        nodes = [n for n in nodereader][1:]
    node_names = [n[0] for n in nodes]

    with open(filename, 'r') as edgecsv:
        edgereader = csv.reader(edgecsv)
        edges = [tuple(e) for e in edgereader][1:]

    return node_names, edges


def make_graph(node_names, edges):
    G = nx.Graph()
    G.add_nodes_from(node_names)
    G.add_edges_from(edges)
    return G


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # load data, make graph
        # filename = 'facebook_combined.txt'
        filename = 'twitter_combined.txt'
        # filename = 'twitter_combined_chunk.txt'
        
        # filename = 'facebook_combined_chunk.txt'
        # node_names, edges = load_data(filename)
        # graph = make_graph(node_names, edges)

        # Read the file into a graph
        graph = nx.read_edgelist(filename, delimiter=" ")

        # Serialize the graph
        serialized_graph = pickle.dumps(graph)
    else:
        serialized_graph = None

    # Broadcasting the serialized graph
    serialized_graph = comm.bcast(serialized_graph, root=0)

    # Deserialize the graph on all processes
    graph = pickle.loads(serialized_graph)

    # Distribute nodes among processes
    assigned_nodes = distribute_nodes(graph.nodes(), rank, size)

    # Each process processes its assigned nodes
    partial_sum = process_data(graph, assigned_nodes, rank)

    # Gather the partial results from all processes
    all_res = comm.gather(partial_sum, root=0)

    if rank == 0:
        concatenated_result = {}
        for d in all_res:
            concatenated_result.update(d)
        # print("Concatenated Result:", concatenated_result)
        print('Run complete.')


if __name__ == "__main__":
    main()
