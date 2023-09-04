from collections import defaultdict


class Graph:

    # Constructor
    def __init__(self):

        # Default dictionary to store graph
        self.graph = defaultdict(list)

    # Function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    # Function to print a BFS of graph
    def BFS(self, s):
        flag = True
        if s == 0:
            flag = False
        # Mark all the vertices as not visited
        visited = [False] * (max(self.graph) + 1)

        # Create a queue for BFS
        queue = [s]

        # Mark the source node as
        # visited and enqueue it
        visited[s] = True
        while queue:

            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            # if flag:
            #     print(s)

            # Get all adjacent vertices of the
            # dequeued vertex s.
            # If an adjacent has not been visited,
            # then mark it visited and enqueue it
            for i in self.graph[s]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

        return visited

    def check_if_Bipartite_BFS(self, node):

        visited = [False] * (max(self.graph) + 1)
        sign = [0] * (max(self.graph) + 1)
        queue = [node]
        non_compatible = 0
        visited[node] = True
        sign[node] = 1

        while queue:
            node = queue.pop(0)
            # print(node, end=" ")
            for i in self.graph[node]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True
                    sign[i] = -1 * sign[node]
                else:
                    if sign[i] != -1 * sign[node]:
                        non_compatible += 1

        if non_compatible == 0:
            print("connected component fully bipartite")

        return non_compatible, sign, visited
