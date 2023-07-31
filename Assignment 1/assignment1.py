# FIT2004 Assignment 1
# Name: Loh Jing Wei
# Student ID: 30856183

import math

# ============================================================= Problem 1 ===============================================================

class MinHeap():
    # Referred to Week 12 FIT1008 
    """ 
    ---------------------------------------- A Vertex Class for MinHeap -------------------------------------------------
    The min heap object implement a priority queue by storing the vertices (as node of the heap) in an array. 
    The vertex with shorter travel time from start has the higher priority, in other word, the vertex of the parent 
    node has travel time from start to be less than that of the child node. Hence the vertex at root node must be 
    the vertex with shortest travel time from start among all nodes in heap.

    Attributes:
        vertex_heap:                an array which stores vertices 
        length:                     length of vertex_heap
        index_arr:                  an array which stores the index of each node in vertex_heap, it is use for index mapping.
    """

    def __init__(self, size):
        """
        Description: 
            Constructor of min heap.
            Initialise an empty heap and empty index array for index mapping.
            Initialise the length of the heap to 0.
        Input:
            size: size of heap needed
        Time complexity: 
            Best and Worst: O(N) + O(N) = O(N), where N is number of nodes in heap
        Aux space complexity: 
            O(N) + O(N) = O(N), where N is number of nodes in heap
        """
        self.length = 0
        self.vertex_heap = [None] * (size+1) # empty heap - Time complexity: O(N), aux space: O(N)  
        self.index_arr = [None] * (size) # index array - Time complexity: O(N), aux space: O(N) 

    def append(self,vertex):
        """
        Description: 
            Append vertex object at the end of heap and update the index mapping.
            The function calls rise() operation to rise appended node to its correct position.
        Input:
            vertex: Vertex object
        Time complexity: 
            Best: O(1)
            Worst: O(logN) where N is number of nodes in heap
        Aux space complexity: 
            O(1)
        """
        self.length += 1 # increment length of heap
        self.vertex_heap[self.length] = vertex # append vertex at end of heap
        self.index_arr[vertex.id] = self.length # update index mapping
        self.rise(self.length) # Time complexity: O(logN)

    def __len__(self):
        """
        Description: 
            Returns length of heap.
        Output: length of heap
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        return self.length

    def swap(self, index1, index2):
        """
        Description: 
            Swap 2 nodes in heap and update index mapping.
        Input:
            index1: the index of node to be swapped in heap
            index2: the index of node to be swapped in heap
        Time complexity: 
            Best and Worst: O(1) 
        Aux space complexity: 
            O(1)
        """
        # update index mapping
        self.index_arr[self.vertex_heap[index1].id] = index2
        self.index_arr[self.vertex_heap[index2].id] = index1

        # swap vertex in heap
        vertex1 = self.vertex_heap[index1]
        self.vertex_heap[index1] = self.vertex_heap[index2]
        self.vertex_heap[index2] = vertex1

    def rise(self, k):
        """
        Description: 
            Rise a node at index k to its correct position.
        Input:
            k: the index of node to rise
        Time complexity: 
            Best: O(1) 
            Worst: O(logN), where N is number of nodes in heap
        Aux space complexity: 
            O(1)
        """
        # Time complexity: O(log V)
        while k > 1 and self.vertex_heap[k].travel_time < self.vertex_heap[k // 2].travel_time: # Time complexity: O(LogN)
            self.swap(k, k // 2) 
            k = k // 2

    def smallest_child(self, k):
        """
        Description: 
            returns index of smallest child of node at index k.
        Input:
            k: index of node to find its child
        Output: index of smallest child of node at index k
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        if 2 * k == self.length or self.vertex_heap[2 * k].travel_time < self.vertex_heap[2 * k + 1].travel_time:
            return 2*k
        else:
            return 2*k+1

    def sink(self, k):
        """
        Description: 
            Sink a node at index k to its correct position
        Input:
            k: the index of node to sink
        Time complexity: 
            Best: O(1) 
            Worst: O(logN) where N is number of nodes in heap
        Aux space complexity: 
            O(1)
        """
        while 2*k <= self.length: # Time complexity: O(logN)
            child = self.smallest_child(k)
            if self.vertex_heap[k].travel_time <= self.vertex_heap[child].travel_time:
                break
            self.swap(child, k)
            k = child

    def serve(self):
        """
        Description: 
            Function which returns the smallest node (vertex with smallest travel time from source).
        Output: smallest node
        Time complexity: 
            Best: O(1)
            Worst: O(logN) where N is number of nodes in heap
        Aux space complexity: 
            O(1)
        """
        if self.length>0:
            # swap minimum node with last node
            self.swap(1, self.length)
            # remove the min node from index mapping and heap
            self.index_arr[self.vertex_heap[self.length].id] = None
            min_vertex = self.vertex_heap[self.length]
            self.length -= 1
            # sink the swapped node to correct position
            self.sink(1) # Time complexity: O(logN)
            return min_vertex
        else:
            return "Heap is empty"

    # diff from 1008
    def update_travel_time(self, vertex, new_travel_time):
        """
        Description: 
            Updates travel time of vertex object.
        Input:
            vertex: Vertex object to be updated with new travel time from source
            new_travel time: new travel time for vertex from source
        Time complexity: 
            Best: O(1)
            Worst: O(logN) where N is number of nodes in heap
        Aux space complexity: 
            O(1)
        """
        # update vertex travel time
        vertex.travel_time = new_travel_time
        # extract index of vertex in heap from index mapping array
        index = self.index_arr[vertex.id]
        # rise updated vertex node to correct position
        self.rise(index) # Time complexity: O(logN)

class Graph: 
    """ 
    -------------------------------------------- A Vertex Class for Graphs --------------------------------------------------
    The Graph object is an adjacency matrix which stores a list of vertices (locations), and each vertex has a list of edges (roads). 
    The Graph object is essentially a two layered graph, where the first layer contains the original vertices 
    that are given in the input, and each of the Vertex has a list of edges with the alone lane travel time as its weight. 
    The second layer contains identical vertices, with different id, that has the carpool lane travel time 
    as the edges weight instead.
    If the Vertex has passenger to be pick up as stated in passengers list, an Edge with zero travel time is created
    to connect this Vertex with it's corresponding second layer Vertex. 

    Attributes:
        max_vertex:                 An integer value for the id of vertex with largest id.
        vertices_count:             An integer value representing number of vertices (locations) from input
        layered_vertices_count:     An integer value for total number of vertices in layered graph
        vertices_arr:               An array of vertices in layered graph
    """

    def __init__(self,roads, passengers):
        """
        Description: 
            Constructor for Graph object.
        Input:
            roads: a list of tuples with 4 integers (a,b,c,d) where
                a is starting location of road
                b is ending location of road
                c is travel time of the road when travelling alone (using alone lane)
                d is travel time of the road when travelling with passengers (using carpool lane)
            passengers: list of location (vertex) with passengers that can be pick up
        Time complexity: 
            Best and Worst: O(E) + O(2V) + O(2V) + O(E) = O(E+V), where E is number of edges and V is number of vertices
        Aux space complexity: 
            O(2V) = O(V), where V is number of vertices
        """
        # Find count of vertices from input roads
        self.max_vertex = roads[0][0]
        for i in range(len(roads)): # Time complexity: O(E)
            if roads[i][0] > self.max_vertex:
                self.max_vertex = roads[i][0]
            elif roads[i][1] > self.max_vertex:
                self.max_vertex = roads[i][1]
        self.vertices_count = (self.max_vertex + 1)
        self.layered_vertices_count = self.vertices_count * 2
        
        # Initialise array of vertex
        self.vertices_arr = [None]* (self.layered_vertices_count) # Time complexity: O(2V), Aux space: O(2V)
        for i in range(self.layered_vertices_count): # Time complexity: O(2V)
            if i<self.vertices_count:
                self.vertices_arr[i] = Vertex(i,i)
            else:
                self.vertices_arr[i]= Vertex(i,i-self.vertices_count)

        # Add edges to Vertex
        self.add_edges(roads,passengers) # Time complexity: O(E)
    
    def add_edges(self,roads,passengers):
        """
        Description: 
            Insert edges from roads list to vertices' list of edges.
        Input:
            roads: a list of tuples (a,b,c,d) where
                a is starting location of road
                b is ending location of road
                c is travel time of the road when travelling alone (using alone lane)
                d is travel time of the road when travelling with passengers (using carpool lane)
            passengers: list of location (vertex id) with passengers that can be pick up
        Time complexity: 
            Best and Worst: O(E), where E is number of edges 
        Aux space complexity: 
            O(1)
        """
        for edge in roads: # Time complexity: O(E)
            # Edge for alone lane
            u = edge[0]
            v = edge[1]
            w = edge[2] # alone lane travel time

            # add edge to u
            current_edge = Edge(u,v,w)
            current_vertex = self.vertices_arr[u]
            current_vertex.add_edge(current_edge)

            # Edge for carpool lane
            carpool_u = u + self.vertices_count 
            carpool_v = v + self.vertices_count
            carpool_w = edge[3] # carpool lane travel time

            # create edge with zero travel time to travel from first layer vertex to the identical second layer vertex
            # when there is passengers
            if u in passengers:
                current_vertex.add_edge(Edge(u,carpool_u,0))

            # add edge to carpool_u
            current_carpool_edge = Edge(carpool_u,carpool_v,carpool_w)
            current_carpool_vertex = self.vertices_arr[carpool_u]
            current_carpool_vertex.add_edge(current_carpool_edge)

    def djikstra(self, start_vertex_id):
        """
        Description: 

            Djikstra finds shortest travel time from departure location (start vertex) 
            for each vertices (location points) in Graph object.

        Approach:

            The function calls constructor of Minheap and append the each vertex into the heap, 
            using the heap as priority queue for vertices.
            Starting from the start vertex (in the first layer), Djikstra iteratively use serve function of MinHeap to select 
            unvisited vertex with the shortest travel time from start vertex. It then discover vertices which has an 
            incoming edge from this vertex and, if path with shorter travel time is found, update the travel time using
            update_travel_time function of MinHeap. The process is continues until all reachable vertices (in both layers) has been visited.

        Input:
            start_vertex_id: the start vertex id (the starting location)

        Precondition: start_vertex_id is an integer, it correspond to the id attribute of the start Vertex object
        Postcondition:  All Vertex objects in the Graph object connected to the start Vertex will have attribute distance 
                        updated with the shortest distance from start.

        Time complexity: 
            Best and Worst: O(V) + O(VlogV) + O(logV) + O(VlogV) + O(V^2logV) = O(ElogV), where E is number of edges and V is number of vertices
                - in worst case where the graph is dense, each vertex has V-1 edges, and in this case V^2 can be written as E.
        
        Aux space complexity: 
            O(V) for MinHeap, where V is number of vertices

        """
        # Initialise MinHeap for vertex
        djikstra_heap = MinHeap(self.layered_vertices_count) # Time complexity: O(V), Aux space: O(V)

        # Append vertex into MinHeap
        for vertex in self.vertices_arr: # O(V)
            djikstra_heap.append(vertex) # Time complexity: O(logV)

        # Discover the start vertex and update the travel time to 0
        start_vertex = self.vertices_arr[start_vertex_id]
        start_vertex.discovered = True
        djikstra_heap.update_travel_time(start_vertex,0) # Time complexity: O(logV)

        # iteratively visit vertex with the shortest travel time from start vertex
        while djikstra_heap.length > 0: # Time complexity: O(V)
            u = djikstra_heap.serve() # Time complexity: O(logV)

            # If no routes from source to vertex u
            if not u.discovered:
                continue

            u.visited = True

            # discover vertices with incoming edges from the visited vertex and update distance accordingly
            for edge in u.edges: # Time complexity: O(V), for dense graph each vertex has V-1 edges
                v = self.vertices_arr[edge.v]
                v_newdist = u.travel_time + edge.w
                
                if not v.discovered:
                    v.discovered = True
                    djikstra_heap.update_travel_time(v,v_newdist) # Time complexity: O(logV)
                    v.previous = u

                elif not v.visited:
                    if v.travel_time > v_newdist:
                        djikstra_heap.update_travel_time(v,v_newdist) # Time complexity: O(logV)
                        v.previous = u

    def __str__(self):
        return_string = ""
        for vertex in self.vertices_arr:
            return_string = return_string + "Vertex " + str(vertex) + "\n"
        return return_string
    
class Vertex:
    """ 
    --------------------------------------------- A Vertex Class for Vertex -----------------------------------------------------
    The Vertex object is used to represent the Locations. 

    Attributes:
        id:                 A unique id to distinguish vertices apart
        print_id:           An id which is the same for the identical vertices in layered graph
        edges:              A list of outgoing Edges (roads) of Vertex
        discovered:         A boolean value which states if the Vertex has been discovered
        visited:            A boolean value which states if the Vertex has been visited
        travel_time:        An ineteger value for the travel time of the Vertex from start location
        previous:           The previous Vertex (use for backtracking the path from start)
    """

    def __init__(self,id,print_id):
        """
        Description: 
            Constructor for Vertex object.
        Input:
            id: A unique id to distinguish vertices apart
            print_id: An id which is the same for the identical vertices in layered graph
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        self.id = id
        self.print_id = print_id
        self.edges = []
        # for traversal
        self.discovered = False
        self.visited = False
        # travel time of vertex from start
        self.travel_time = math.inf
        # for backtracking 
        self.previous = None

    def add_edge(self,edge):
        """
        Description: 
            Append Edge object into list of edges.
        Input:
            edge: Edge object (outgoing edge from the Vertex)
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        self.edges.append(edge)
     
    def __str__(self):
        return_string = str(self.id)
        for edge in self.edges:
            return_string = return_string + "\n with edges: " + str(edge) 
        return_string = return_string + "\n travel time: " + str(self.travel_time)
        if self.previous != None:
            return_string = return_string + "\n previous: " + str(self.previous.id)
        return return_string

class Edge:
    """ 
    ----------------------------------------------- A Vertex Class for Edge ----------------------------------------------------
    The Edge object is used to represent the Road. 

    Attributes:
        u:      Integer value representing the id of starting Vertex (location) of the Edge (road)
        v:      Integer value representing the id of ending Vertex (location) of the Edge (road)
        w:      Integer value representing the travel time on the Edge (road)
    """

    def __init__(self,u,v,w):
        """
        Description: 
            Constructor for Vertex object.
        Input:
            u: Integer value representing the id of starting Vertex (location) of the Edge (road)
            v: Integer value representing the id of ending Vertex (location) of the Edge (road)
            w: Integer value representing the travel time on the Edge (road)
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        self.u = u
        self.v = v
        self.w = w
    
    def __str__(self):
        return_string = str(self.u) + "," + str(self.v) + "," + str(self.w)
        return return_string
    
def optimalRoute(start,end,passengers,roads):
    """

    Description: 

        optimalRoute finds the path from start to end with shortest travel time. 
        It returns a list representing the path with shortest travel time from start to end.
 
    Approach:

        The function construct a Graph object for the input list of roads. 
        Each location will be represented as Vertex objects and each road as Edge object in the Graph.

        Every location will initialised 2 Vertex object, one for each layer of the Graph.
        Both the Vertex object has list of edges representing the outgoing roads from it. 
        The edges are identical for both vertex except:
            - First layer Vertex has Edge attribute 'w' representing the travel time using alone lane on the road
            - Second layer Vertex has Edge attribute 'w' representing the travel time using carpool lane on the same road

        By using layered graph approach, the Djikstra function will only need to be called once to compute 
        the shortest travel time for travelling from start to end location in both scenarios:
            1. using only alone lane and 
            2. when carpool lane is used (when passenger is pick up).
        This reduces the time complexity as Djikstra function does not need to be called twice for both scenarios.

        The function compare the travel time to the end location from the start between using only alone lane and 
        when carpool lane is used, the function then choose the path with shortest travel time. 
            - This can be done by comparing the attribute 'travel_time' for both Vertex object of the end location.
        
        By backtracking using the attribute 'previous' of Vertex object, the function return a list of 
        locations for the path with shortest travel time from start to end.

    Input:
        start: the departure location 
        end: the destination location
        passengers: list containing integers representing locations which has passengers to be pick up
        roads: a list of tuples (a,b,c,d) where
            a is starting location of road
            b is ending location of road
            c is travel time of the road when travelling alone (using alone lane)
            d is travel time of the road when travelling with passengers (using carpool lane)

    Output: 
        ret_list:   a list of integer representing the path with shortest travel time from start to end
                    
    Time complexity: 
        Best and Worst: O(R+L) + O(RlogL) + O(L) + O(L) = O(|R|log|L|), where R is the number of roads and L is number of locations
        
    Aux space complexity: 
        O(|R|+|L|)

    """
    # empty return list to store the path later
    ret_list = [] 

    # Initialise Graph object
    layered_graph = Graph(roads, passengers) # Time complexity: O(R+L)

    # Call djikstra function
    layered_graph.djikstra(start) # Time complexity: O(RlogL)

    # Compare the travel time between 2 scenarios: using only alone lane and when carpool lane is use
    end_vertex = layered_graph.vertices_arr[end]
    end_vertex_with_passenger = layered_graph.vertices_arr[layered_graph.vertices_count+end]

    if end_vertex.travel_time <= end_vertex_with_passenger.travel_time:
        ret_list.append(end_vertex.print_id)
        vertex = end_vertex
    else:
        ret_list.append(end_vertex_with_passenger.print_id)
        vertex = end_vertex_with_passenger

    # Backtrack from the end vertex - Append the locations of the path with shortest travel time into return list
    while vertex.previous != None: # Time complexity: O(L)
        if vertex.previous.print_id != vertex.print_id:
            ret_list.append(vertex.previous.print_id)
        vertex = vertex.previous
    
    # Reverse the path (initially path has end location at the beginning of the list)
    ret_list.reverse() # Time complexity: O(L)
    
    return ret_list # Aux space: O(R)

# ============================================================= Problem 2 ===============================================================
class Section:
    """ 
    ----------------------------------------------- A Vertex Class for Section ----------------------------------------------------
    The Section object is used to represent one single section of the staff office.

    Attributes:
        index:                                  Tuple value (i,j), represents the location of Section
                                                    where i is an integer value referring to row index
                                                    where j is an integer value referring to column index
        accumulated_occupancy_probability:      Integer value representing the accumulated ccupancy probability of the Section
        previous:                               The previous Section object (use for backtracking)
    """

    def __init__(self,index):
        """
        Description: 
            Constructor for Section object.
        Input:
            index: Tuple value (i,j), represents the location of Section
                        where i is an integer value referring to row index
                        where j is an integer value referring to column index
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        self.index = index
        self.accumulated_occupancy_probability = 0
        self.previous = None

    def __str__(self):
        return_str = "" + str(self.accumulated_occupancy_probability)
        return return_str
    
def select_sections(occupancy_probability):
    """

    Description: 

        select_sections find the n sections with lowest total occupancy probability to be removed,
        where one section must be remove from each rows and removal in two adjacent rows must be in same or adjacent columns.
 
    Approach:

        The function first construct a memo matrix of n+1 rows and m columns, 
        where n and m is the number of rows and number of columns of input occupancy_probability respectively.

        The last row of the memo will be use as base case, 
        and each Section object in the last row will have the attribute accumulated_occupancy_probability updated as 0.

        The recurrence relationship of the function can be written as:
            when i = n (last row), memo[i][j] = 0.
            when m == 1 (there is only a single column), memo[i][j] = occupancy_probability[i][j] + memo[i+1][j] (same column)
            when j = m-1 (last column), memo[i][j] = occupancy_probability[i][j] 
                                                    + min(  memo[i+1][j-1] (left adjacent column), 
                                                            memo[i+1][j] (same column),
            when j = 0 (first column), memo[i][j] = occupancy_probability[i][j] 
                                                    + min(  memo[i+1][j] (same column),
                                                            memo[i+1][j+1](right adjacent column))
            otherwise, memo[i][j] = occupancy_probability[i][j] 
                                    + min(  memo[i+1][j-1] (left adjacent column), 
                                            memo[i+1][j] (same column),
                                            memo[i+1][j+1](right adjacent column))
        
        The memo matrix will then be filled from bottom (the second last row) to top, left to right iteratively.
        Each memo[i][j] will store a newly initialised Section object.
        When filling up the memo, the function will always find the minimum accumulated occupancy probability 
        of the row below it, then update the current Section object's accumulated_occupancy_probability attribute as  
        sum of the minimum accumulated occupancy probability of the row below with the current Section occupancy probability 
        as given in the input occupancy_probability matrix (occupancy_probability[i][j]).

        From the memo, find the Section with the least accumulated_occupancy_probability from the first row.
        The accumulated_occupancy_probability found here is the minimum_total_occupancy.

        The function then backtrack for the n sections resulting in the minimum_total_occupancy using Section object attribute previous. 
        The previous attribute in Section object stores the Section object from the row below which contributes 
        to the accumulated probability of the current Section.
        This allows the function to backtrack until it reaches the base case, where the previous attribute is None.

    Input:
        occupancy_probability:      list of list of integer, where integer at occupancy_probability[i][j] represent 
                                    the occupancy probability of a section at row i column j.

    Output:
        ret_list:   list of 2 item:
                        minimum_total_occupancy: an integer which is total occupancy for the n sections selected
                        sections_location:  list of n tuples in the form of (i,j), 
                                            each tuple represent the location of a single section to be removed from top to bottom

    Time complexity: 
        Best and Worst: O(n) +O(n) + O(nm) + O(m) + O(n) = O(mn), where n is number of rows and m is number of columns/aisle
        
    Aux space complexity: 
        O(nm) for the memo matrix + O(n) + O(n) = O(nm), where n is number of rows and m is number of columns/aisle

    """
    # number of rows
    n = len(occupancy_probability) 
    # number of columns
    m = len(occupancy_probability[0])

    # Initialise memo matrix
    memo = [None]*(n+1) # Time complexity: O(n), Aux space: O(nm) after for loop
    for i in range(n+1): # Time complexity: O(n)
        if i == n:
            memo[i] = [Section((i,j)) for j in range(m)] # last row as base case
        else:
            memo[i] = [None]*m 

    # fill up the memo matrix from bottom (second last row) to top
    for i in range(n-1,-1,-1): # Time complexity: O(nm)
        for j in range(m):
            memo[i][j] = Section((i,j))
            if m == 1: # input only has a single column 
                accumulated_probability = occupancy_probability[i][j] + memo[i+1][j].accumulated_occupancy_probability
                memo[i][j].accumulated_occupancy_probability = accumulated_probability
                memo[i][j].previous = memo[i+1][j]
            elif j==0: # first column
                # find the min accumulated occupancy probability between same column vs right adjacent column of the row below
                min_prob = select_min_accumulated_probability(memo[i+1][j],memo[i+1][j+1]) 
                accumulated_probability = occupancy_probability[i][j] + min_prob.accumulated_occupancy_probability
                memo[i][j].accumulated_occupancy_probability = accumulated_probability
                memo[i][j].previous = min_prob
            elif j==(m-1): # last column
                # find the min accumulated occupancy probability between left adjacent column vs same column of the row below
                min_prob = select_min_accumulated_probability(memo[i+1][j],memo[i+1][j-1]) 
                accumulated_probability = occupancy_probability[i][j] + min_prob.accumulated_occupancy_probability
                memo[i][j].accumulated_occupancy_probability = accumulated_probability
                memo[i][j].previous = min_prob            
            else:
                # find the min accumulated occupancy probability between left adjacent column 
                # vs same column vs right adjacent column of the row above
                min_prob = select_min_accumulated_probability(memo[i+1][j],memo[i+1][j-1],memo[i+1][j+1])
                accumulated_probability = occupancy_probability[i][j] + min_prob.accumulated_occupancy_probability
                memo[i][j].accumulated_occupancy_probability = accumulated_probability
                memo[i][j].previous = min_prob
    
    # Search in the first row of memo for Section object with least accumulated_occupancy_probability
    # The accumulated_occupancy_probability found is the minimum_total_occupancy to be return
    minimum_total_occupancy = math.inf
    min_section = None
    for i in range(m): # Time complexity: O(m)
        if memo[0][i].accumulated_occupancy_probability < minimum_total_occupancy:
            min_section = memo[0][i]
            minimum_total_occupancy = min_section.accumulated_occupancy_probability

    # Backtracking to find the n sections to be removed
    sections_location = [] # aux space: O(n) after loop
    sections_location.append(min_section.index)
    current = min_section.previous
    while current.previous != None: # Time complexity: O(n)
        sections_location.append(current.index)
        current = current.previous

    ret_list = [minimum_total_occupancy,sections_location] # aux space: O(n)

    return ret_list

def select_min_accumulated_probability(section1,section2,section3=None):
    """
    Description: 
        Helper function which returns the Section object with least accumulated_occupancy_probability.
    Input:
        section1: a Section object
        section2: a Section object
        section3: a Section object
    Output:
        Section object with least accumulated_occupancy_probability.
    Time complexity: 
        Best and Worst: O(1)
    Aux space complexity: 
        O(1)
    """
    if section3 != None:
        if section1.accumulated_occupancy_probability <= section2.accumulated_occupancy_probability and section1.accumulated_occupancy_probability <= section3.accumulated_occupancy_probability:
            return section1
        elif section2.accumulated_occupancy_probability <= section3.accumulated_occupancy_probability:
            return section2
        else:
            return section3
    else:
        if section1.accumulated_occupancy_probability <= section2.accumulated_occupancy_probability:
            return section1
        else:
            return section2


