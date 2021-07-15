# THIS IS FUNCTIONAL!
# https://www.sthu.org/code/codesnippets/imagepers.html
# Uses topology to determine peaks. 
# From there, I need to filter peaks depending on the area somehow?
# It shows peaks when diffuse! 

# I Think I need to write custom version that can take into account no peaks WITHIN other peaks for prominence

# UPDATE 6-24-2021: Tried using a median filter around each peak to see if I can filter out non-peak like peaks from the topology algo. 
# Unfortunately, I wasn't able to find a window that could distinguish. May be other methods

# Next attempt: create a minimum spanning tree, and find the center point of each line segment. The 
# average value of these is likely to be a good representation of the "cell background". 
# 1. Calculate a minimum spanning tree
# 2. Calculate a mean for all points in tree, and "centerpoint mean" for each edge
# 2. If a peak is closer to the mean than it is to the centerpoint mean, then the peak is barely a peak. 

import random 
from collections import deque

from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from findpeaks import findpeaks
from tqdm import tqdm

class Coordinate:
    """ Class for storing coordinate data. Easy to get the distance"""
    def __init__(self, x, y, value):
        self.x = int(x)
        self.y = int(y)
        self.value = value
    def __repr__(self):
        return f"({self.x},{self.y})"

    def __sub__(self, other):
        return (self.x - other.x)**2 + (self.y - other.y)**2
        # No need to square root to compare distances.

    def __lt__(self, other):
        in1 = self.x + self.y
        in2 = other.x + other.y
        if in1<in2:
            return True
        else:
            return False
    
    def __gt__(self, other):
        in1 = self.x + self.y
        in2 = other.x + other.y
        if in1 > in2:
            return True
        else:
            return False
    
    def min(self, other):
        # "Minimum point" for making it easy to calculate edges
        if self.x != other.x:
            if self.x < other.x:
                return self
            else:
                return other
        else:
            if self.y < other.y:
                return self
            elif self.y > other.y:
                return other
            else:
                return None

class Edge:
    def __init__(self, point_a, point_b):
        self.a = point_a
        self.b = point_b
        self.value = point_a-point_b
    
    def __iter__(self):
        return iter((self.a,self.b))

    def __repr__(self):
        return f"({self.a},{self.b})"
    
    def __eq__(self, other):
        if isinstance(other, Edge):
            if self.a == other.a and self.b == other.b:
                return True
            elif self.a == other.b and self.b == other.a:
                return True
        return False

    def __hash__(self):
        return hash(min(self.a,self.b))

class Graph:
#https://www.techiedelight.com/check-undirected-graph-contains-cycle-not/

# Constructor
    def __init__(self, edges):

        # A list of lists to represent an adjacency list
        #self.adjList = [[] for _ in range(N)]
        self.adj_list = {}

        # add edges to the undirected graph
        #print("GRAPH EDGES: ",edges)
        for src, dest in edges:
            if not list(self.adj_list.keys()).count(src):
                self.adj_list[src] = []
            if not list(self.adj_list.keys()).count(dest):
                self.adj_list[dest]= []
            self.adj_list[src].append(dest)
            self.adj_list[dest].append(src)

    # https://www.geeksforgeeks.org/union-find/
    def find_parent(self, parent,i):
        if parent[i] == -1:
            return i
        if parent[i]!= -1:
             return self.find_parent(parent,parent[i])
 
    # A utility function to do union of two subsets
    def union(self,parent,x,y):
        parent[x] = y

    # The main function to check whether a given graph
    # contains cycle or not
    def isCyclic(self):
         
        # Allocate memory for creating V subsets and
        # Initialize all subsets as single element sets
        parent = [-1]*(len(self.adj_list.keys()))
 
        # Iterate through all edges of graph, find subset of both
        # vertices of every edge, if both subsets are same, then
        # there is cycle in graph.
        for i in self.graph:
            for j in self.graph[i]:
                x = self.find_parent(parent, i)
                y = self.find_parent(parent, j)
                if x == y:
                    return True
                self.union(parent,x,y)

def BFS(graph, src):
    # to keep track of whether a vertex is discovered or not
    discovered = {}
    # mark the source vertex as discovered
    discovered[src] = True
    # create a queue for doing BFS
    q = deque()
    # enqueue source vertex and its parent info
    q.append((src, -1))
    # loop till queue is empty
    while q:
        # dequeue front node and print it
        (v, parent) = q.popleft()
        # do for every edge `v —> u`
        #print(f"Checking adjacent to {v}: {graph.adj_list[v]}")
        for u in graph.adj_list[v]:
            if not list(discovered.keys()).count(u):
                # mark it as discovered
                discovered[u] = True
                # construct the queue node containing info
                # about vertex and enqueue it
                q.append((u, v))
            # `u` is discovered, and `u` is not a parent
            elif u != parent:
                # we found a cross-edge, i.e., the cycle is found
                return True
    # no cross-edges were found in the graph
    return False

def label_peaks(data):
    """ 
    This is the main peak labeling function. Uses the findpeaks library to determine significant peaks
    in an image. The algorithm works REALLY well at labeling peaks when phase separated, but does not 
    filter out the diffuse frames at all. 

    Input: raw image data (single frame)
    Output: a list of slices that contain the topological peaks. 
    """

    # Use findpeaks to label the peaks via topological prominence
    fp = findpeaks(method='topology', window=3)
    results = fp.fit(data)

    # Create an array of the peaks labeled for numpy interaction
    maxima = np.array(results['Xdetect'])
    #unique, counts = np.unique(maxima, return_counts=True)
    labeled, num_objects = ndimage.label(maxima)

    # Slice the array to get the coordinates of the labeled pixels. 
    sliced = {i:list(zip(*np.where(labeled==i))) for i in np.unique(labeled) if i}
    return sliced
    

def filter_peaks(slices, data):
    """ Reads in the peaks, and determines whether a peak is significant enough (non-phase separated maxima still register as peaks) """
    
    #xs, ys = [], []
    coord_list = []
    
    for i,coords in enumerate(slices.values()):
        for y,x in coords:
            # Arbitrary filtration of peaks that are not above background
            if data[y][x] > 200:
                #print("")
                #xs.append(x)
                #ys.append(y)
                coord_list.append(Coordinate(x,y,data[y][x]))
    
    #print(coord_list)
    ######
    #coord_list = [Coordinate(170,84,0), Coordinate(172,85,0),Coordinate(169,81,0)]
    ######
    edges = krus_mst(coord_list)
    #subXs, subYs = bresenham_line(xs[1],ys[1],xs[2],ys[2])
    return coord_list, edges

def prim_mst(coords):
    mst_verts = []
    edges = []
    # put first vertex into the mst set
    mst_verts.append(coords.pop(0))
    # Get the nearest vertex to the first vertex
    while coords:
        # LOOP THROUGH ALL MST POINTS
        curr = mst_verts[random.randint(0,len(mst_verts)-1)]
        #print(f"MST SET: {mst_verts}")
        next_vert = nearest(coords,curr)
        print(curr)
        print(next_vert)

        edges.append(Edge(curr,next_vert))
        mst_verts.append(coords.pop(coords.index(next_vert)))
    return edges

def krus_mst(coords):
    edges = {}
    mst_verts = []
    mst_edges = []

    for coord in coords:
        for other_coord in coords:
            if coord != other_coord:
                edge = Edge(coord,other_coord)
                edges[edge]=edge.value
    edges = dict(sorted(edges.items(), key=lambda item: item[1]))
    origin = next(iter(edges)).a
    #print({A:N for (A,N) in [x for x in edges.items()][:4]})
    #while edges:
    while len(mst_edges) < (len(coords)-1):
        next_edge = next(iter(edges))
        print("NEXT EDGE:",next_edge)
        if mst_edges:
            if not mst_edges.count(next_edge):    
                mst_edges.append(next_edge)
                print("Creating graph")
                graph = Graph(mst_edges)
                if graph.isCyclic():
                    print("Cyclic, not using edge")
                    mst_edges.remove(next_edge)
                    edges.pop(next_edge)
                else:
                    print("Acyclic! Adding edge.")
                    edges.pop(next_edge)
            else:
                print("Duplicate detected!")
                edges.pop(next_edge)
        else:
            mst_edges.append(next_edge)
            edges.pop(next_edge)
    print(f"Found {len(mst_edges)} edges for {len(coords)} vertices")
    print(mst_edges)
    return mst_edges
        


        # If vertices have not been added yet, add them. 
        #if (not mst_verts.count(next_edge.a) and not mst_verts.count(next_edge.b)):
            #mst_verts.append(next_edge.a)
            #mst_verts.append(next_edge.b)
            #mst_edges.append(next_edge)
            #edges.pop(next_edge)
            #print("Added edge", next_edge)
        #elif (not mst_verts.append(next_edge.a) and mst_verts.count(next_edge.b)):
            # pathfinding
        #    print("needs pathfinding a to b")
        #elif ( mst_verts.append(next_edge.a) and not mst_verts.count(next_edge.b)):
        #    print("needs pathfinding b to a")
            # pathfinding


def nearest(coords, coord):
    """Returns the vertex that is nearest to the starting vertex"""
    minimum = None
    min_dist = 1000000000
    for vert in coords:
        dist = coord - vert
        if dist < min_dist:
            min_dist = dist
            minimum = vert
    return minimum 

def bresenham_line(x0, y0, x1, y1):
    # https://stackoverflow.com/questions/50995499/generating-pixel-values-of-line-connecting-2-points
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0  
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1: 
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []    
    for x in range(x0, x1 + 1):
        if steep:
            line.append((y,x))
        else:
            line.append((x,y))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    print(f"From ({x0},{y0}) to ({x1},{y1}) is {line}")
    print(f"Centerpoint is {line[int(len(line) / 2)]}")
    return line


#img = Image.open('data/test_data-14bit.tif')
img = Image.open('data/FullScale_BGsub.tif')
nframes = range(img.n_frames)[0:1]
cycle = 0
for i,frame in tqdm(enumerate(nframes)):
    img.seek(frame)
    data = np.array(img)
    # Gauss filter to smooth out the clipped peaks
    gauss = ndimage.filters.gaussian_filter(data,sigma=1)
    slices = label_peaks(gauss)
    peaks, edges = filter_peaks(slices,gauss)
    #print("PEAKS!",peaks)
    xs = []
    ys = []

    for peak in peaks:
        xs.append(peak.x)
        ys.append(peak.y)
    #print(xs)


    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = axes.ravel()
    ax[cycle].imshow(data)
    ax[cycle].axis('off')
    cycle+=1
    ax[cycle].imshow(data)
    ax[cycle].axis('off')
    ax[cycle].plot(xs,ys, 'r.')
    lines = []
    for edge in edges:
        #print(f"({edge.a.x}, {edge.a.y}); ({edge.b.x},{edge.b.y})")
        lines.append([(edge.a.x,edge.a.y),(edge.b.x,edge.b.y)])
    lc = mc.LineCollection(lines, colors="orange", linewidths=0.5)
    ax[cycle].add_collection(lc)
        #ax[cycle].axline((edge.a.x,edge.a.y),(edge.b.x,edge.b.y))
    #for j in range(len(xs)):
    #    plt.annotate(j, (xs[j], ys[j]))
    cycle+=1
    plt.autoscale(False)
    plt.savefig(f'result-{i}.png', bbox_inches = 'tight')
    cycle=0
    plt.show()
    plt.close(fig)

