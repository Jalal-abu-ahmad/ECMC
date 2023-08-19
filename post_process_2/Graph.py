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

        # Mark all the vertices as not visited
        visited = [False] * (max(self.graph) + 1)

        # Create a queue for BFS
        queue = []

        # Mark the source node as
        # visited and enqueue it
        visited[s] = True

        while queue:

            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            print(s, end=" ")

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
        sign = 0 * (max(self.graph) + 1)
        local_sign = 1
        queue = []
        non_compatible = 0
        visited[node] = True
        sign[node] = local_sign

        while queue:
            node = queue.pop(0)
            print(node, end=" ")
            for i in self.graph[node]:
                local_sign *= -1
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True
                    sign[i] = local_sign
                else:
                    if sign[i] != local_sign:
                        non_compatible += 1
                        print("saaaaaaaaaaaaaaaaaaaaaaaaaaad")

        return non_compatible
