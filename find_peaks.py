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
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from findpeaks import findpeaks
from tqdm import tqdm
from collections import defaultdict

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

# Constructor
    def __init__(self, edges):
        #print("Creating graph")
        self.graph = defaultdict(list)

        i = 0
        for src, dest in edges:
            #print("ADDING EDGE",src,dest)
            if src.key is None:
                src.key = i 
                i += 1
            if dest.key is None:
                dest.key = i
                i += 1
            self.add_edge(src,dest)
        self.verts = i
        
        #    if src not in temp:
        #        temp[src] = []
        #    if dest not in temp:
        #        temp[dest] = []
        #    if dest not in temp[src]:
        #        temp[src].append(dest)
        #    if src not in temp[dest]:
        #        temp[dest].append(src)
        #print(temp)
        #self.graph = temp
        #for src_ in temp.keys():
        ##    for dest_ in temp[src_]:
         #       self.graph[src_].append(dest_)
         #       self.graph[dest_].append(src_)
            #if not list(self.adj_list.keys()).count(src):
            #    self.adj_list[src] = []
            #if not list(self.adj_list.keys()).count(dest):
            #    self.adj_list[dest]= []
            #self.adj_list[src].append(dest)
            #self.adj_list[dest].append(src)
        #self.verts = len(self.graph.keys())
        #print(f"there are {self.verts} verts and {len(edges)} edges: \n {self.graph}")

    def add_edge(self,u,v):
        self.graph[u].append(v)

    # https://www.geeksforgeeks.org/union-find/
    def find_parent(self, parent,i):
        #print("FINDING PARENT", parent, i)
        if parent[i] == -1:
            return i
        if parent[i]!= -1:
             return self.find_parent(parent,parent[i])
 
    def union(self,parent,x,y):
        parent[x] = y

    def isCyclic(self):
         
        parent = [-1]*(self.verts)
        for i in self.graph:
            for j in self.graph[i]:
                x = self.find_parent(parent, i)
                y = self.find_parent(parent, j)
                if x == y:
                    return True
                self.union(parent,x,y)
        return False

        #for m,i in enumerate(self.graph):
            #print("I",i)
       #     for n,j in enumerate(self.graph[m]):
        #        #print("J",j)
         #       x = self.find_parent(parent, i)
          #      y = self.find_parent(parent, j)
           #     print("X AND Y:", x,y)
            #    if x == y:
             #       return True
              #  self.union(parent,x,y)

    
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
                coord_list.append(Coordinate(x,y,data[y][x],i))
    edges = krus_mst(coord_list)
    
    print(f"Starting with {len(coord_list)} peaks")
    for edge in edges:
        mid = edge.mid()
        #print(mid)
        #print(edge.a,edge.b,mid,mid.x,mid.y)
        #print(data[edge.a.y][edge.a.x],data[edge.b.y][edge.b.x],data[mid.y][mid.x])
        p1 = data[edge.a.y][edge.a.x]
        p2 = data[edge.b.y][edge.b.x]
        p3 = data[mid.y][mid.x]
        remove = False
        if ((p1+p2)/2-p3)/p3 > 0.4:
            if p1 < p3:
                if edge.a in coord_list:
                    coord_list.remove(edge.a)
                    remove = True
            if p2 < p3:
                if edge.b in coord_list:
                    coord_list.remove(edge.b)
                    remove = True
            if remove:
                edges.remove(edge)
        else:
            if edge.a in coord_list:
                    coord_list.remove(edge.a)
                    remove = True
            if edge.b in coord_list:
                    coord_list.remove(edge.b)
                    remove = True
            if remove:
                edges.remove(edge)
            

    print(f"Ended with {len(coord_list)} peaks")
    #subXs, subYs = bresenham_line(xs[1],ys[1],xs[2],ys[2])
    return coord_list, edges

def krus_mst(coords):
    edges = {}
    mst_edges = []

    # Start by building a dictionary of ALL possible edges (it is immense) and getting their weights, for sorting. 
    for coord in coords:
        for other_coord in coords:
            if coord != other_coord:
                edge = Edge(coord,other_coord)
                edges[edge]=edge.value
    edges = dict(sorted(edges.items(), key=lambda item: item[1]))

    # Build the graph. 
    while len(mst_edges) < (len(coords)-1):
        print(f"HAVE {len(mst_edges)} edges out of {len(coords)-1}")
        next_edge = next(iter(edges))
        if mst_edges:
            for edge in mst_edges:
                edge.a.key = None
                edge.b.key = None
            # If the first edge has been added, try adding a new edge and seeing if it 
            # forms a cyclic graph. If it does not, then keep it, otherwise remove it. 
            if not mst_edges.count(next_edge):    
                # If this edge (forward or back) is not already in the graph...
                mst_edges.append(next_edge)
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
            # First cycle: just add the first edge of lowest distance. 
            mst_edges.append(next_edge)
            edges.pop(next_edge)

    print(f"Found {len(mst_edges)} edges for {len(coords)} vertices")
    #print(mst_edges)
    return mst_edges




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
    print("PEAKS!",peaks)
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

