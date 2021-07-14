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

from PIL import Image
import numpy as np
from numpy.lib import imag
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from findpeaks import findpeaks
from tqdm import tqdm

import networkx as nx

class Coordinate:
    """ Class for storing coordinate data. Easy to get the distance"""
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value
    def __repr__(self):
        return f"({self.x},{self.y})"

    def __sub__(self, other):
        return (self.x - other.x)**2 + (self.y - other.y)**2
        # No need to square root to compare distances.

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
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.value = a-b
    
    def __repr__(self):
        return f"[{self.a},{self.b}:{self.value}]"

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

                #xs.append(x)
                #ys.append(y)
                coord_list.append(Coordinate(x,y,data[y][x]))
                
    
    
    print(coord_list)
    edges = prim_mst(coord_list.copy())


    #subXs, subYs = bresenham_line(xs[1],ys[1],xs[2],ys[2])
    return coord_list, edges

def prim_mst(coords):
    mst_verts = []
    edges = []
    # put first vertex into the mst set
    mst_verts.append(coords.pop(0))
    # Get the nearest vertex to the first vertex
    while coords:
        curr = mst_verts[-1]
        #print(f"MST SET: {mst_verts}")
        next_vert = nearest(coords,curr)
        edges.append(Edge(curr,next_vert))
        mst_verts.append(coords.pop(coords.index(next_vert)))
    return edges


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
    print("PEAKS!",peaks)
    xs = []
    ys = []

    for peak in peaks:
        xs.append(peak.x)
        ys.append(peak.y)
    print(xs)


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
        print(f"({edge.a.x}, {edge.a.y}); ({edge.b.x},{edge.b.y})")
        lines.append([(edge.a.x,edge.a.y),(edge.b.x,edge.b.y)])
    lc = mc.LineCollection(lines, colors="orange")
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

