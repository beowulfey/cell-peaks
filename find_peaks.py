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

#from . import union_find 

from PIL import Image
import numpy as np
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from findpeaks import findpeaks
from tqdm import tqdm
from collections import defaultdict
from statistics import NormalDist

class Coordinate:
    """ Class for storing coordinate data. Easy to get the distance"""
    def __init__(self, x, y, value=0,key=None):
        self.x = x
        self.y = y
        self.value = value
        self.key = None
    def __index__(self):
        return self.key

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

class Edge:
    def __init__(self, point_a, point_b):
        self.a = point_a
        self.b = point_b
        self.value = point_a-point_b
    
    def bline(self):
    # https://stackoverflow.com/questions/50995499/generating-pixel-values-of-line-connecting-2-points
        x0 = self.a.x
        x1 = self.b.x
        y0 = self.a.y
        y1 = self.b.y
        
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
                line.append(Coordinate(y,x))
            else:
                line.append(Coordinate(x,y))

            error = error + deltay
            if error > 0:
                y = y + ystep
                error = error - deltax
        if switched:
            line.reverse()
        #print(f"From ({x0},{y0}) to ({x1},{y1}) is {line}")
        #print(f"Centerpoint is {line[round(len(line) / 2)]}")
        return line
    
    def mid(self):
        line = self.bline()
        mid = line[round(len(line) / 2)]
        return mid
    
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
    """
    Graph class for generating the minimum spanning tree. 

    The graph is generated by taking a list of edges, each of which has two endpoints.
    The endpoints (Coordinates) have a key that is reset in the function that calls this (should I do it on init? Probably)
    This key is used for counting the number of vertices, as well as for uniqueness purposes.

    The purpose of this graph is to determine whether there are any cyclic segments; if none is found, the version of the graph
    is kept and continues to grow as I find the MSP. 

    Input: A list of edges.
    Purpose: Determine whether the edges are cyclic. 

    """ 
    # Sources:
    # https://www.geeksforgeeks.org/union-find/

# Constructor
    def __init__(self, edges):
        self.graph = defaultdict(list)
        # I should probably reset the keys used below here. 
        i = 0
        for src, dest in edges:
        # Go through each edge and annotate each coordinate with a unique key (without duplication)
            if src.key is None:
                src.key = i 
                i += 1
            if dest.key is None:
                dest.key = i
                i += 1
            self.add_edge(src,dest)
        self.verts = i
        
    def add_edge(self,u,v):
        self.graph[u].append(v)
    def find_parent(self, parent,i):
        if parent[i] == -1:
            return i
        if parent[i]!= -1:
             return self.find_parent(parent,parent[i])
    def union(self,parent,x,y):
        parent[x] = y

    def isCyclic(self):
        # The main function called by the program. Uses the above functions. 
        parent = [-1]*(self.verts)
        for i in self.graph:
            for j in self.graph[i]:
                x = self.find_parent(parent, i)
                y = self.find_parent(parent, j)
                if x == y:
                    return True
                self.union(parent,x,y)
        return False

    
def confidence(data,confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1+confidence)/ 2.)
    h = dist.stdev * z/ ((len(data)-1) ** 0.5)
    return h

def label_peaks(data):
    """ 
    This is the main peak labeling function. Uses the findpeaks library to determine significant peaks
    in an image. The algorithm works REALLY well at labeling peaks when phase separated, but does not 
    filter out the diffuse frames at all. 

    Input: raw image data (single frame)
    Output: a list of slices that contain the topological peaks. 
    """

    # Use findpeaks to label the peaks via topological prominence
    fp = findpeaks(method='topology', cu='0.6', window=5)
    results = fp.fit(data)

    # Create an array of the peaks labeled for numpy interaction
    maxima = np.array(results['Xdetect'])
    #unique, counts = np.unique(maxima, return_counts=True)
    labeled, num_objects = ndimage.label(maxima)

    # Slice the array to get the coordinates of the labeled pixels. 
    sliced = {i:list(zip(*np.where(labeled==i))) for i in np.unique(labeled) if i}
    return sliced
    

def filter_peaks(slices, data):
    """ 
    Reads in the peaks, and determines whether a peak is significant enough 
    (non-phase separated maxima still register as peaks)
    """
    
    coord_list = []
    
    for i,coords in enumerate(slices.values()):
    # Start by converting the slices into a usable format.  
        for y,x in coords:
            # Arbitrary filtration of peaks that are not above background
            if data[y][x] > 500:
                coord_list.append(Coordinate(x,y,data[y][x],i))
    
    # Build the Minimum Spanning Tree with Kruskal's algorithm 
    edges = krus_mst(coord_list)

    p1s = []
    p2s = []
    p3s = []
    mids = []

    for edge in edges:
        mid = edge.mid()
        mids.append(mid)
        p1s.append(data[edge.a.y][edge.a.x])
        p2s.append(data[edge.b.y][edge.b.x])
        p3s.append(data[mid.y][mid.x])

    #stdev = int(np.std([i for n, i in enumerate(p1s) if i not in p1s[:n]] + [j for m, j in enumerate(p2s) if j not in p2s[:m]]))
    stdev = int(np.std(p3s))
    cf = confidence(p3s, confidence = 0.95)
    meanmid = int(np.mean(p3s))
    #print(f"Mean of midpoints is {meanmid} and upper of peaks is {meanmid+cf}")
    removed = []
    for n,edge in enumerate(edges):
        midv = p3s[n]
        #print(f"EDGE: {edge} -> {edge.a.value}, {edge.b.value}, {midv} vs {meanmid} +/- {cf}")
        remove = False
        if cf + midv > p1s[n]-cf:
            if edge.a in coord_list:
                    coord_list.remove(edge.a)
                    remove = True
        if cf + midv > p2s[n]-cf:
            if edge.b in coord_list:
                    coord_list.remove(edge.b)
                    remove = True
        if remove:
            removed.append(edge)

    if removed:
        [edges.remove(edge) for edge in removed]


    print(f"Ended with {len(coord_list)} peaks")
    return coord_list, edges, mids

def krus_mst(coords):
    """
    Function to determine the minimum spanning tree of the detected peaks. 

    Uses Kruskal's algorithm to determine the minimum spanning tree. Given a dictionary of all 
    possible edges from the included coordinates, sort them by shortest to longestand continually 
    add a new edge. Each time, the current state of the graph is checked to see whether
    it forms a cycle. If it does, the edge is removed and the graph building continues.

    Continues until there are N-1 edges, given N coordinates. 

    Input: a list of Coordinates
    Output: a list of edges that form the minimum spanning tree. 
    """
    edges = {}
    mst_edges = []

    # Start by building a dictionary of ALL possible edges (it is immense) and getting their weights, for sorting. 
    for coord in coords:
        for other_coord in coords:
            if coord != other_coord:
                edge = Edge(coord,other_coord)
                edges[edge]=edge.value
    # sort the dictionary
    edges = dict(sorted(edges.items(), key=lambda item: item[1]))

    # Build the graph. 
    while len(mst_edges) < (len(coords)-1):
        #print(f"HAVE {len(mst_edges)} edges out of {len(coords)-1}")
        next_edge = next(iter(edges))
        if mst_edges:
        # If the first edge has been added, try adding a new edge and seeing if it 
        # forms a cyclic graph. If it does not, then keep it, otherwise remove it. 
            for edge in mst_edges:
            # Reset the keys for each coordinate (these are later generated during the 
            # graph formation)
                edge.a.key = None
                edge.b.key = None
            if not mst_edges.count(next_edge):    
                # If this edge (forward or back) is not already in the graph...
                mst_edges.append(next_edge)
                graph = Graph(mst_edges)
                if graph.isCyclic():
                    #print("Cyclic, not using edge")
                    mst_edges.remove(next_edge)
                    edges.pop(next_edge)
                else:
                    #print("Acyclic! Adding edge.")
                    edges.pop(next_edge)
            else:
                #print("Duplicate detected!")
                edges.pop(next_edge)
        else:
            # First cycle: just add the first edge of lowest distance. 
            mst_edges.append(next_edge)
            edges.pop(next_edge)

    #print(f"Found {len(mst_edges)} edges for {len(coords)} vertices")
    return mst_edges


# Main Sequence (should extract into MAIN)

#img = Image.open('data/test_data-14bit.tif')
img = Image.open('data/FullScale_BGsub.tif')
#img = Image.open('data/AVG_TC-olaIs39.tif')

x2s = []
y2s = []
nframes = range(img.n_frames)
#cycle = 0
for i,frame in tqdm(enumerate(nframes)):
    print(f"FRAME {i}")
    img.seek(frame)
    data = np.array(img)

    # Gauss filter to smooth out the clipped peaks
    gauss = ndimage.filters.gaussian_filter(data,sigma=1)
    #gauss = data
    slices = label_peaks(gauss)
    peaks, edges, mids = filter_peaks(slices,gauss)
    
    xs = []
    ys = []
    x2s.append(i/4)
    y2s.append(len(peaks))
    
    
    #print("=====================")
    #print("PEAK\tVALUE")
    #print("=====================")
    for peak in peaks:
    #    print(f"{peak}      {peak.value}")
        xs.append(peak.x)
        ys.append(peak.y)
    #fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,2),(0,0))
    plt.imshow(data)
    ax2 = plt.subplot2grid((2,2),(0,1))
    plt.imshow(gauss)
    plt.plot(xs,ys, 'r.',markersize='2')
    ax3 = plt.subplot2grid((2,2),(1,0), colspan=2)
    axes = plt.gca()
    axes.set_xlim([0,9])
    axes.set_ylim([0,70])
    axes.set_xlabel("Time (min)")
    axes.set_ylabel("Number of condensates")
    plt.plot(x2s,y2s, color='red')
    if i>=8:
        plt.vlines(8/4,0,50,colors='gray',linestyles='dashed')
        plt.text(x=8/4-0.5,y=55, s="KCl added")

    #ax[cycle].plot(x2s,y2s, 'g.')
    
    # Draw minimum spanning tree
    #lines = []
    #for edge in edges:
        #print(f"({edge.a.x}, {edge.a.y}); ({edge.b.x},{edge.b.y})")
    #    lines.append([(edge.a.x,edge.a.y),(edge.b.x,edge.b.y)])
    #lc = mc.LineCollection(lines, colors="blue", linewidths=0.5)
    #ax[cycle].add_collection(lc)
        #ax[cycle].axline((edge.a.x,edge.a.y),(edge.b.x,edge.b.y))
    #for j in range(len(xs)):
    #    plt.annotate(j, (xs[j], ys[j]))
    #cycle+=1

    #plt.autoscale(False)
    plt.savefig(f'result-{i}.png', bbox_inches = 'tight',dpi=300)
    #cycle=0
    #plt.show()
    plt.close(fig)

