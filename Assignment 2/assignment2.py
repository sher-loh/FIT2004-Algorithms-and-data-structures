# FIT2004 Assignment 2
# Name: Loh Jing Wei
# Student ID: 30856183

import math
from collections import deque


class Vertex:
    """ 
    ----------------------------------------------- Class for Vertex ----------------------------------------------------
    The Vertex object is used to represent the data centres. 

    Attributes:
        id:                 A unique id to distinguish vertices apart
        edges:              A list of outgoing Edges (connection channels) of Vertex
        discovered:         A boolean value which states if the Vertex has been discovered
        visited:            A boolean value which states if the Vertex has been visited
        min_flow:           An integer value for the minimum flow of the path which pass through this Vertex
        previous:           The previous Edge (use for backtracking the path from source)

    """
    def __init__(self,id):
        """
        Description: 
            Constructor for Vertex object.
        Input:
            id: A unique id to distinguish vertices apart
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        self.id = id
        self.edges = []
        # for traversal
        self.discovered = False
        self.visited = False
        # travel time of vertex from start
        self.min_flow = math.inf
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
     
class Edge:
    """ 
    ----------------------------------------------- Class for Edge ----------------------------------------------------
    The Edge object is used to represent the connection channels and maximum amount of outgoing and incoming data 
    that data centre i can process per second. 

    Attributes:
        u:              Integer value representing the id of starting Vertex of the Edge 
        v:              Integer value representing the id of ending Vertex of the Edge
        residual:       Integer value representing the residual (remaining capacity) on the Edge
        reverse_edge    Edge object, the edge in the opposite direction that is connected to same two Vertex.
                        i.e. Edge connect from v to u       
    """
    def __init__(self,u,v,residual):
        """
        Description: 
            Constructor for Vertex object.
        Input:
            u:              Integer value representing the id of starting Vertex of the Edge 
            v:              Integer value representing the id of ending Vertex of the Edge
            residual:       Integer value representing the residual (remaining capacity) on the Edge
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            O(1)
        """
        self.u = u
        self.v = v
        self.residual = residual
        self.reverse_edge = None


class ResidualNetwork: 
    """ 
    ----------------------------------------------- Class for Residual Network ------------------------------------------------------------
    The ResidualNetwork object represent data centres and connections between them.

    Each data centre i is represented as two Vertex objects, x and y, which is connected by an Edge object from x to y:
        first vertex (vertex x) receive data from other data centres
        second vertex (vertex y) outputs data to other daata centres
    the residual of Edge between this two Vertex represent:
        Source data centre: maximum amount of outgoing data that data centre i can process per second
        Target data centres: maximum amount of incoming data that data centre i can process per second
        Other data centres: maximum amount of outgoing data or maximum amount of incoming data, whichever that is smaller.

    Each communication channels going from data centres u to data centres v is represented as Edge object. 
    The residual (remaining capacity) of the Edge represent the remaining throughput of the channel, 
    it is initialised as the maximum throughput of the channel.
    The edge connects from second vertex of data centre u to first vertex of data centre v.
    
    Each Edge object has a reverse Edge object with 0 residual, it is use for flow that can be cancelled.

    An additional Vertex call super sink is created for multiple targets, and an Edge object connects the 
    second Vertex of each target data centre to super sink Vertex.

    Attributes:
        data_centres_count:         An integer value, represent number of data centres.
        vertices_count:             An integer value, represent number of vertices needed to represent data centres for residual network.
                                    It is the data_centres_count*2+1.
                                        double of data_centres_count to allow edge within each data centre.
                                        one additional vertex for super sink.
        super_sink:                 An integer value, represent the id of super sink vertex.
        source:                     An integer value, represent the id of source vertex (origin data centre).
        vertices_arr:               A list of Vertex object (data centres) of length vertices_count.
                                    Index                       Item
                                    i                           first Vertex of data centre i
                                    i+data_centres_count        second Vertex of data centre i 
                                    -1                          super sink Vertex
                                    where i is the id of data centre

    """

    def __init__(self, connections, maxIn, maxOut, origin, targets):
        """
        Description: 
            Constructor of ResidualNetwork.
            Initialise a list of Vertex object and Edge between vertices.
        Input:
            connections:    list of tuple with 3 integers (a,b,t) where
                                a is ID of the data centre from which the communication channel departs
                                b is ID of the data centre to which the communication channel arrives
                                t is positive integer representing the maximum throughput of that channel
            maxIn:          list of integers represent maximum amount of incoming data that data centre i can process per second.
            maxOut:         list of integers represent maximum amount of outgoing data that data centre i can process per second.
            origin:         an integer value which represent the ID of the data centre where the data to be backed up is located.
            targets:        list of integers represent the ID of data centres backup data to be stored.
        Time complexity: 
            Best and Worst: O(D+C), where D is number of data centres and C is number of connections
        Aux space complexity: 
            O(D) for vertices_arr, where D is number of data centres
        """
        # Count of data centres
        self.data_centres_count = len(maxIn)
        # Count of vertices needed
        self.vertices_count = self.data_centres_count*2 + 1 

        # index of super sink 
        self.super_sink = self.vertices_count-1
        self.source = origin
   
        # Initialise array of Vertex object
        self.vertices_arr = [None]* (self.vertices_count) # Time complexity: O(2D), Aux space: O(2D)
        for i in range(self.vertices_count): # Time complexity: O(2D)
                self.vertices_arr[i] = Vertex(i)

        # Add edges to Vertex
        self.add_edges(connections) # Time complexity: O(C)
        self.add_max_edge(targets,maxIn,maxOut) # Time complexity: O(D)
        self.add_edge_to_super_sink(targets, maxIn) # Time complexity: O(D)
    
    def add_edges(self,connections):
        """
        Description: 
            Insert edges from connections list to each Vertex's list of edges. 
            Each Edge object represent the communication channels from data centres u to data centres v.
                The edge connects from second vertex of data centre u to first vertex of data centre v.
                The residual attribute of Edge is the maximum throughput of the channel.
            For every Edge object inserted to Vertex u, a reverse Edge object with residual 0 will be inserted to Vertex v.
        Input:
            connections:    list of tuple with 3 integers (a,b,t) where
                                a is ID of the data centre from which the communication channel departs
                                b is ID of the data centre to which the communication channel arrives
                                t is positive integer representing the maximum throughput of that channel
        Time complexity: 
            Best and Worst: O(C), where C is the number of connection channels
        Aux space complexity: in-place, O(1)
        """
        for edge in connections: # Time complexity: O(C)
            u = edge[0] + self.data_centres_count # index of second Vertex of data centre from which the communication channel departs
            v = edge[1] # index of first Vertex of data centre from which the communication channel arrives
            residual = edge[2] # maximum throughput 

            # add edge to u
            current_edge = Edge(u,v,residual)
            current_vertex = self.vertices_arr[u]
            current_vertex.add_edge(current_edge)

            # add reverse edge to v
            current_reverse_edge = Edge(v,u,0)
            current_vertex = self.vertices_arr[v]
            current_vertex.add_edge(current_reverse_edge)        

            # Edge's reverse edge is each other
            current_edge.reverse_edge = current_reverse_edge
            current_reverse_edge.reverse_edge = current_edge
    
    def add_max_edge(self,targets,maxIn,maxOut):
        """
        Description: 
            Insert edges between first Vertex and second Vertex of every data centre according to maxIn and maxOut
            the residual of Edge between this two Vertex represent:
                Source data centre: maximum amount of outgoing data that data centre i can process per second (maxOut)
                Target data centres: maximum amount of incoming data that data centre i can process per second (maxIn)
                Other data centres: maximum amount of outgoing data (maxOut) or maximum amount of incoming data (maxIn), whichever that is smaller.
        Input:
            maxIn:          list of integers represent maximum amount of incoming data that data centre i can process per second.
            maxOut:         list of integers represent maximum amount of outgoing data that data centre i can process per second.
            targets:        list of integers represent the ID of data centres backup data to be stored.
        Time complexity: 
            Best and Worst: O(D), where D is the number of data centres
        Aux space complexity: 
            in-place, O(1)
        """
        for i in range(self.data_centres_count): # Time complexity: O(D)
            u = i # ID of first vertex of data centre i
            intermediate_u = i+self.data_centres_count # ID of second vertex of data centre i

            # For source data centre -> always choose maxOut because no incoming data (no need to consider maxIn)
            if i == self.source: 
                max_edge = Edge(u,intermediate_u,maxOut[u])
                u_vertex = self.vertices_arr[u]
                u_vertex.add_edge(max_edge)
            
            # For target data centres -> always choose maxIn because no outgoing data (no need to consider maxOut)
            elif i in targets: 
                max_edge = Edge(u,intermediate_u,maxIn[u])
                u_vertex = self.vertices_arr[u]
                u_vertex.add_edge(max_edge)

            # for other data centres -> choose the smaller value between maxIn and maxOut
            elif maxIn[u] <= maxOut[u]:
                # add maxIn edge from first vertex to second vertex 
                max_edge = Edge(u,intermediate_u,maxIn[u])
                u_vertex = self.vertices_arr[u]
                u_vertex.add_edge(max_edge)
            else:
                # add maxOut edge from first vertex to second vertex 
                max_edge = Edge(u,intermediate_u,maxOut[u])
                u_vertex = self.vertices_arr[u]
                u_vertex.add_edge(max_edge)

            # add reverse edge from intermediate u to u vertex
            max_reverse_edge = Edge(intermediate_u,u,0)
            intermediate_u_vertex = self.vertices_arr[intermediate_u]
            intermediate_u_vertex.add_edge(max_reverse_edge)

            # Edge's reverse edge is each other
            max_edge.reverse_edge = max_reverse_edge
            max_reverse_edge.reverse_edge = max_edge


    def add_edge_to_super_sink(self, targets, maxIn):
        """
        Description: 
            Insert edges from second Vertex of targets data centre to super sink Vertex.
        Input:
            targets:        list of integers represent the ID of data centres backup data to be stored.
            maxIn:          list of integers represent maximum amount of incoming data that data centre i can process per second.
        Time complexity: 
            Best and Worst: O(D) where D is number of data centres
        Aux space complexity: 
            in-place, O(1)
        """
        super_sink = self.super_sink
        super_sink_vertex = self.vertices_arr[super_sink]

        for target in targets: # Time complexity: O(D)
            intermediate_target = target+self.data_centres_count
            target_vertex = self.vertices_arr[intermediate_target] 

            # add edge from second vertex of target to super sink
            target_to_sink_edge = Edge(intermediate_target, super_sink, maxIn[target])
            target_vertex.add_edge(target_to_sink_edge)

            # add reverse edge from super sink to second vertex of target
            sink_to_target_edge = Edge(super_sink,intermediate_target,0)
            super_sink_vertex.add_edge(sink_to_target_edge)

            # Edge's reverse edge is each other
            target_to_sink_edge.reverse_edge = sink_to_target_edge
            sink_to_target_edge.reverse_edge = target_to_sink_edge

    def bfs(self):
        """
        Description: 
            Breadth First Search to find path from source to sink. 
        Output:
            True when there is path from source to sink
            else False when there is no path from source to sink.
        Time complexity: 
            Best: O(D) where D is number of data centres
            Worst: O(D+C) where D is number of data centres and C is number of connections
        Aux space complexity: 
            O(D) where D is number of data centres
        """
        # reset vertices
        for v in self.vertices_arr: # Time complexity: O(D)
            v.discovered, v.visited, v.previous, v.min_flow = False, False, None, math.inf

        source = self.vertices_arr[self.source]
        discovered_arr = deque([]) 
        discovered_arr.append(source)

        while len(discovered_arr) > 0: # Time complexity: O(D+C)
            u = discovered_arr.popleft()
            u.visited = True
            
            # reach super sink and return true, end BFS early
            if u.id == self.super_sink:
                return True
            
            for edge in u.edges:
                v = self.vertices_arr[edge.v] #retrieve vertex object
                if v.discovered == False and v.visited == False and edge.residual != 0:
                    discovered_arr.append(v)
                    v.discovered = True
                    v.previous = edge
                    v.min_flow = min(edge.residual,u.min_flow)

        # no more path to super sink, return false
        return False

# ======================================================== Q1 Code =================================================================
def ford_fulkerson(connections, maxIn, maxOut, origin, targets):
    """
    Description: 
        Ford-fulkerson method to find the maximum flow in network.
    Input: 
        connections:    list of tuple with 3 integers (a,b,t) where
                            a is ID of the data centre from which the communication channel departs
                            b is ID of the data centre to which the communication channel arrives
                            t is positive integer representing the maximum throughput of that channel
        maxIn:          list of integers represent maximum amount of incoming data that data centre i can process per second.
        maxOut:         list of integers represent maximum amount of outgoing data that data centre i can process per second.
        origin:         an integer value which represent the ID of the data centre where the data to be backed up is located.
        targets:        list of integers represent the ID of data centres backup data to be stored.
    Output:
        max_flow:       an integer value, represent the maximum flow possible in Residual network
    Time complexity: 
        Best: O(D+C)
        Worst: O(FD+FC) = O(DC^2) where D is number of data centres, C is number of connections and F is flow
    Aux space complexity: 
        in-place O(1)
    """
    #initialise flow of the network
    max_flow = 0
    #initialise residual network
    residual_network = ResidualNetwork(connections, maxIn, maxOut, origin, targets) # Time complexity: O(D+C)
    #while there is augmenting path 
    while residual_network.bfs(): # Time complexity: O(FD+FC)
        current_vertex = residual_network.vertices_arr[residual_network.super_sink]
        path_min_flow = current_vertex.min_flow

        # Augment flow - Backtracking to update residual of edges 
        while current_vertex.id != residual_network.source: # Time complexity: O(D)
            previous_edge = current_vertex.previous
            reverse_edge = previous_edge.reverse_edge

            previous_edge.residual -= path_min_flow
            reverse_edge.residual += path_min_flow

            current_vertex = residual_network.vertices_arr[current_vertex.previous.u]

        # add the min flow of the path to flow of network
        max_flow += path_min_flow

    return max_flow
    
def maxThroughput(connections, maxIn, maxOut, origin, targets):
    """
    Description: 
        maxThroughput function runs Ford-fulkerson method to find the maximum possible 
        data throughput from the data centre origin to the data centres specified in targets.
    Approach:
        Ford-fulkerson method:
            Initialise a Residual Network for the data centres and connection channel.
            Path Augmentation:
                Find the paths traversal from source to sink in the residual network using BFS.
                Use the minimum flow of path found to augment the flow:
                    Update residual of edges.
                    Update the maximum flow of network.
                Repeat until no more path from source to sink.
            The maximum flow found for the Residual network is the maximum possible data throughput 
            from the data centre origin to the data centres specified in targets.
    Input:
        connections:    list of tuple with 3 integers (a,b,t) where
                            a is ID of the data centre from which the communication channel departs
                            b is ID of the data centre to which the communication channel arrives
                            t is positive integer representing the maximum throughput of that channel
        maxIn:          list of integers represent maximum amount of incoming data that data centre i can process per second.
        maxOut:         list of integers represent maximum amount of outgoing data that data centre i can process per second.
        origin:         an integer value which represent the ID of the data centre where the data to be backed up is located.
        targets:        list of integers represent the ID of data centres backup data to be stored.
    Output:
        max_flow:       an integer value, represent the maximum possible data throughput
                        from the data centre origin to the data centres specified in targets.
    Time complexity: 
        Best: O(D+C) where D is number of data centres, C is number of connections 
        Worst: O(DC^2) where D is number of data centres, C is number of connections 
    Aux space complexity: 
        in-place, O(1)
    """
    max_flow = ford_fulkerson(connections, maxIn, maxOut, origin, targets)
    return max_flow


# ======================================================== Q2 Code =================================================================

class Node:
    """ 
    ----------------------------------------------- Class for Node ----------------------------------------------------
    The Node object for Trie class.

    Attributes:
        link:               A list which stores the connected child Nodes. 
                            The index of the link list represents the next alphabet of a sentence.
                                index + 97 - 1 = Ascii of character 
        frequency:          An integer value which represents the frequency of most frequent completed sentence,  
                            this sentence has prefix up until current node. (this sentence begins with string value up until current node)
        next_index:         Node                next_index
                            Terminal node       None (no connected child node, sentence is complexted)
                            Other ndoe          An integer value which represents the index of the next node in the current link list that 
                                                leads to the completed sentence with the highest frequency.
                                                (it also represents the next alphabet of this sentence)
        data:               Node                data
                            Terminal node       A string value, represents the completed sentence in the terminal node. 
                            other node          None
    
    """
    def __init__(self):
        """
        Description: 
            Constructor for Node object.
        Input:
            size:   int value for the size of link list, it is the number of unique possible char in sentence.
                    default value is 27 for [a...z] and $ (terminal)
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            in-place, O(1)
        """
        # size of link list - 27 for [a...z] and $ (terminal)
        self.link = [None]*27 
        # data payload
        self.frequency = 0
        self.next_index = None
        self.data = None
        

# Trie data structure
class CatsTrie:
    """ 
    ----------------------------------------------- Class for CatsTrie ----------------------------------------------------
    CatsTrie object is a prefix Trie for sentences of cats.
    It allows us to insert each sentence into the Trie and 
    auto complete an incomplete sentence according to the frequencies of each sentences used by cats.

    Attributes:
        root:               A Node object for the root of Trie.
    """
    def __init__(self,sentences):
        """
        Description: 
            Constructor for CatsTrie object.
        Approach:
            The init function insert each sentence in sentences list into Trie by calling the insert_recursion() function.
            Every sentence is a string value. Each character of input string is represented by a connection between nodes.
            insert_recursion() function:
                For every string input, insert_recursion function will start from root node and connect nodes to form a chain of nodes.
                    Each chain of nodes with terminal node represent a unique sentence given in input sentences list.
                Recursion:
                    each character of input String is parsed one by one.
                        if the node with connection which represent the current parsed character already exist, continue recursion to next character.
                        else if this node does not exist, new node is created and connected to represent current parsed character, then continue recursion to next character.
                Base case:
                    occurs when no more character can be parse from input String.
                    A terminal node is created if the input String (sentence) has not exist in the Trie.
                        data attribute of the terminal node is updated to the input string (sentence). 
                    Update the frequency attribute of terminal node - it is the number of times this sentence is inputted into Trie 
                    (number of times this sentence is use by a cat)
                At the end of each recursion:
                    update_frequency_and_next_index() function is called for current node (of the recursive level)
                    This function does the below:
                        If the current input string has a higher frequency than value stores in current node frequency attribute 
                                                (stores the frequency of previously most frequently use sentence)
                            -> Update attributes of current node
                                    update the frequency attribute to the current input string's frequency.
                                    update the next_index to the node's index in link list with connection representing next alphabet of input string.
                        else if the frequency is the same
                            -> pick the lexicogrpahically smaller string and update attributes of current node accordingly.
        Input:
            sentences:      a list of string values, each string represent a sentence of cuddly cats.
        Time complexity: 
            Best and Worst: O(NM) where N is the number of sentence in sentences and M is the number of characters in the longest sentence.
        Aux space complexity: 
            O(NM) where N is the number of sentence in sentences and M is the number of characters in the longest sentence (recursive stack).
        """
        self.root = Node()
        current = self.root
        # Call the insert function for every sentence in sentences list 
        for sentence in sentences:
            index_to_child, child_frequency = self.insert_recursion(current,sentence)
            self.update_frequency_and_next_index(current, index_to_child, child_frequency)



    def autoComplete(self,prompt):
        """
        Description: 
            autoComplete function complete the incomplete sentence inputted as prompt, according to the frequency of each sentence use by cat.
            It returns the completed sentence - the most frequently used sentence which begins with the prompt. 
        Approach:
            The function iteratively goes through each char of prompt. 
            For each char, it checks if there is a link from one char to next char in prompt.
            If there is no link between the char -> return None (no sentence begins with this prompt)
            If there is link between char for the whole prompt -> function traverse the node to find terminal node
                The data attribute of this terminal node represent the most frequently used sentence which begins with the prompt
                return this sentence
        Input:
            prompt:         String value, represent the incomplete sentence
        Time complexity: 
            Best: O(X) where X is the length of the prompt. Occurs when sentence with prefix of prompt does not eixst.
            Worst: O(X+Y) where X is the length of the prompt and Y is the length of the most frequent sentence in sentences that begins with the prompt.
        Aux space complexity: 
            in-place, O(1)
        """
        # begin from root
        current = self.root
        return_str = ""
        # go through the char of key 
        for char in prompt: # Time complexity: O(X)
            # find index of char in the list link
            index = ord(char) - 97 + 1
            # if path exist
            if current.link[index] is not None:
                return_str = return_str + char
                current = current.link[index]
            # if path does not exist -> sentence with prefix of prompt does not eixst
            else:
                return None
        
        # go through the chain of node unil the terminal node
        # this chain of nodes represent the incomplete part of node from the most frequently used sentence which begins with the prompt
        while current.next_index is not None: # Time complexity: O(Y)
            current = current.link[current.next_index]
        return current.data
        

    def insert_recursion(self,current,word,i=0):
        """
        Description: 
            insert_recursion function is a helper recursive function for init function.
        Input:
            i:         recursion level, starts from 0
        Time complexity: 
            Best and Worst: O(M) where M is the number of characters in the longest sentence.
        Aux space complexity: 
            O(M) for the recursive stack
        """
        # base case
        if i==len(word):
            index = 0 
            if current.link[index] is not None:
                current = current.link[index]
            else:
                current.link[index] = Node()
                current = current.link[index]
                current.data = word
            # update the frequency of sentence occuring
            current.frequency += 1
            return index,current.frequency

        # recursion
        else:
            # find index of char in the list link
            index = ord(word[i]) - 97 + 1
            # if path exist -> convert ascii to index
            if current.link[index] is not None:
                current = current.link[index]
            # if path does not exist 
            else:
                current.link[index] = Node()
                current = current.link[index]
            # update attributes of current node at the end of recursion
            index_to_child, child_frequency = self.insert_recursion(current,word,i+1)
            self.update_frequency_and_next_index(current, index_to_child, child_frequency)
            return index, current.frequency
        
    def update_frequency_and_next_index(self,current_node, index_to_child, child_frequency):
        """
        Description: 
            update_frequency_and_next_index function is a helper function to update attribute of current node at the end of each recursion.
        Input:
            current_node:           current node at the current recursive level
            index_to_child:         integer value, represent the index of child node in current node's link list.
            child_frequency:        integer value, represent the frequency in which string input occured in Trie.
        Time complexity: 
            Best and Wost: O(1)
        Aux space complexity: 
            in-place, O(1)
        """
        # compare frequency of current node's highest frequency sentence with child node's highest frequency sentence -> select sentence with higher frequency
        if child_frequency > current_node.frequency:
            current_node.frequency = child_frequency
            current_node.next_index = index_to_child
        # if sentence has same frequency -> compare lexicographically -> choose the smaller one
        elif child_frequency == current_node.frequency:
            if index_to_child < current_node.next_index:
                current_node.frequency = child_frequency
                current_node.next_index = index_to_child


