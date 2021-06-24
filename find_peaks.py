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

from operator import itemgetter
from PIL import Image
import numpy as np
from numpy.lib import imag
from scipy import ndimage
import matplotlib.pyplot as plt
from findpeaks import findpeaks
from tqdm import tqdm
import networkx as nx

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return (self.x - other.x)**2 + (self.y - other.y)**2
        # No need to square root to compare distances.


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
    
def filter_peaks(slices):
    """ Reads in the peaks, and determines whether a peak is significant enough (non-phase separated maxima still register as peaks) """
    xs, ys = [], []
    
    for i,coords in enumerate(slices.values()):
        for y,x in coords:
            # Arbitrary filtration of peaks that are not above background
            if gauss[y][x] > 200:

                xs.append(x)
                ys.append(y)
    bresenham_line(xs[0],ys[0],xs[1],ys[1])
    return xs,ys

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
    print(f"From ({y0},{x0}) to ({y1},{x1}) is {line}")
    return line

#img = Image.open('data/test_data-14bit.tif')
img = Image.open('data/FullScale_BGsub.tif')
nframes = range(img.n_frames)[12:13]
cycle = 0
for i,frame in tqdm(enumerate(nframes)):
    img.seek(frame)
    data = np.array(img)
    # Gauss filter to smooth out the clipped peaks
    gauss = ndimage.filters.gaussian_filter(data,sigma=1)
    slices = label_peaks(gauss)
    xs,ys = filter_peaks(slices)
    
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = axes.ravel()
    ax[cycle].imshow(data)
    ax[cycle].axis('off')
    cycle+=1
    ax[cycle].imshow(data)
    ax[cycle].axis('off')
    ax[cycle].plot(xs,ys, 'r.')
    #for j in range(len(xs)):
    #    plt.annotate(j, (xs[j], ys[j]))
    cycle+=1
    plt.autoscale(False)
    plt.savefig(f'result-{i}.png', bbox_inches = 'tight')
    cycle=0
    plt.show()
    plt.close(fig)

