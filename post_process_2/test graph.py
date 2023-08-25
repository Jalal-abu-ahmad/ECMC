import numpy as np
from post_process_2 import Graph


def not_all_visited(visited):

    full_visited = visited[0]
    for i in range(len(visited)):
        full_visited = [full_visited[j] or visited[i][j] for j in range(len(visited[0]))]

    for node in range(len(full_visited)):
        if not full_visited[node]:
            return node
    return -1


g = Graph.Graph()

g.addEdge(0, 1)
g.addEdge(1, 2)
g.addEdge(2, 3)
g.addEdge(3, 0)
g.addEdge(3, 4)
g.addEdge(4, 5)
g.addEdge(2, 6)
g.addEdge(4, 6)

g.addEdge(7, 8)
g.addEdge(7, 9)


i = 0
visited = [g.BFS(0)]
node = not_all_visited(visited)

while node != -1:
    i += 1
    visited.append(g.BFS(node))
    node = not_all_visited(visited)

print("Graph has", len(visited), "connected componentes")


g.check_if_Bipartite_BFS(0)
