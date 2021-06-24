from operator import itemgetter
from PIL import Image
import numpy as np
from numpy.lib import imag
from scipy import ndimage
import matplotlib.pyplot as plt
from findpeaks import findpeaks
from tqdm import tqdm


# THIS IS FUNCTIONAL!
# https://www.sthu.org/code/codesnippets/imagepers.html
# Uses topology to determine peaks. 
# From there, I need to filter peaks depending on the area somehow?
# It shows peaks when diffuse! 

# I Think I need to write custom version that can take into account no peaks WITHIN other peaks for prominence


def label_peaks(data):



    # Gauss filter to smooth out the clipped peaks
    gauss = ndimage.filters.gaussian_filter(data,sigma=1) # 0.75 + window 4 less conservative

    # Use findpeaks to label the peaks via topological prominence
    fp = findpeaks(method='topology', window=4)
    results = fp.fit(gauss)
    #print(results['persistence'])

    # Create an array of the peaks labeled for numpy interaction
    maxima = np.array(results['Xdetect'])
    unique, counts = np.unique(maxima, return_counts=True)
    labeled, num_objects = ndimage.label(maxima)

    # Slice the array to get the coordinates of the labeled pixels. 
    sliced = {i:list(zip(*np.where(labeled==i))) for i in np.unique(labeled) if i}
    xs, ys = [], []
    neighs = []
    median = ndimage.median_filter(data,size=2)
    for i,coords in enumerate(sliced.values()):
        for y,x in coords:
            print(f"{i}: Raw {gauss[y][x]} vs median {median[y][x]} (percent: {int(data[y][x]/median[y][x]*100)}%)")
            if int(data[y][x]/median[y][x]*100) > 120:
                xs.append(x)
                ys.append(y)


    # New stuff:
    # at each label, generate an median pixel value in a window
    
    return gauss,xs,ys,median

img = Image.open('data/test_data-14bit.tif')
nframes = range(img.n_frames)[12:13]
fig, axes = plt.subplots(len(nframes), 3, sharex=True, sharey=True)
cycle = 0
for i,frame in tqdm(enumerate(nframes)):
    ax = axes.ravel()
    img.seek(frame)
    data = np.array(img)
    gauss,xs,ys,median = label_peaks(data)
    
    ax[cycle].imshow(data)                # 0,
    ax[cycle].axis('off')
    cycle+=1
    ax[cycle].imshow(median)
    cycle+=1
    #ax[0].set_title('Original')
    ax[cycle].imshow(gauss)               # 2,3
    ax[cycle].axis('off')
    ax[cycle].plot(xs,ys, 'r.')
    #for j in range(len(xs)):
    #    plt.annotate(j, (xs[j], ys[j]))
    cycle+=1
    plt.autoscale(False)
    plt.savefig(f'result-{i}.png', bbox_inches = 'tight')

plt.show()
