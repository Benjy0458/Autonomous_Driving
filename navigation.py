"""
https://pqdict.readthedocs.io/en/stable/examples.html
"""
import pygraphviz as pgv
from pqdict import minpq
from transpose_dict import TD


def dijkstra(graph, source, target=None):
    dist = {}  # lengths of the shortest paths to each node
    pred = {}  # predecessor node in each shortest path

    # Store distance scores in a priority queue dictionary
    pq = minpq()
    for node in graph:
        if node == source:
            pq[node] = 0
        else:
            pq[node] = float('inf')

    # popitems always pops out the node with min score
    # Removing a node from pqdict is O(log n).
    for node, min_dist in pq.popitems():
        dist[node] = min_dist
        if node == target:
            break

        for neighbor in graph[node]:
            if neighbor in pq:
                new_score = dist[node] + graph[node][neighbor]
                if new_score < pq[neighbor]:
                    # Updating the score of a node is O(log n) using pqdict.
                    pq[neighbor] = new_score
                    pred[neighbor] = node

    return dist, pred


def shortest_path(graph, source, target):
    _, pred = dijkstra(graph, source, target)
    end = target
    path = [end]
    while end != source:
        end = pred[end]
        path.append(end)
    path.reverse()
    return path


if __name__ == '__main__':
    # A simple edge-labeled graph using a dict of dicts
    graph = {
        '0': {'14': 1},
        '1': {'15': 1},
        '2': {},
        '3': {},
        '4': {'18': 1},
        '5': {'19': 1},
        '6': {},
        '7': {},
        '8': {'22': 1},
        '9': {},
        '10': {'24': 1},
        '11': {'25': 1},
        '12': {},
        '13': {},
        '14': {'35': 1},
        '15': {'43': 1},
        '16': {'2': 1},
        '17': {'3': 1},
        '18': {'29': 1},
        '19': {'37': 1},
        '20': {'6': 1},
        '21': {'7': 1},
        '22': {'31': 1},
        '23': {'9': 1},
        '24': {'33': 1},
        '25': {'41': 1},
        '26': {'12': 1},
        '27': {'13': 1},
        '28': {'35': 1},
        '29': {'17': 1, '28': 1},
        '30': {'29': 1},
        '31': {'21': 1, '30': 1},
        '32': {'31': 1},
        '33': {'23': 1, '32': 1},
        '34': {'33': 1},
        '35': {'27': 1, '34': 1},
        '36': {'43': 1},
        '37': {'16': 1, '36': 1},
        '38': {'37': 1},
        '39': {'20': 1, '38': 1},
        '40': {'39': 1},
        '41': {'40': 1},
        '42': {'41': 1},
        '43': {'26': 1, '42': 1}
    }
    # Driving on the left ----
    graph = TD(graph, 1)
    v = {str(i): {} for i in range(44) if str(i) not in graph.keys()}
    graph = v | graph
    # -----

    # dist, pred = dijkstra(graph, source='2')  # Distance from source to all other nodes
    # print(dist), print(pred)

    spawn_points = ['0', '1', '4', '5', '8', '10', '11']
    goal_points = ['2', '3', '6', '7', '9', '12', '13']
    routes = [
        ('0', '13'),
        ('0', '9'),
        ('1', '6'),
        ('1', '2'),
        ('4', '3'),
        ('4', '13'),
        ('5', '6'),
        ('8', '7'),
        ('8', '3'),
        ('10', '9'),
        ('10', '7'),
        ('11', '2'),
        ('11', '12'),
    ]
    # [print(shortest_path(graph, route[0], route[1])) for route in routes]

    # shortest_path = shortest_path(graph, '1', '14')  # List of nodes (inclusive) corresponding to the shortest path
    # print(shortest_path)

    # Draw the graph using pygraphviz
    road_network_graph = pgv.AGraph(graph, strict=True, directed=True)
    road_network_graph.layout(prog='neato')  # Set graph layout
    road_network_graph.draw("graph.svg")  # Draw graph
