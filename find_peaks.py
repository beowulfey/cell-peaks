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
    gauss = ndimage.filters.gaussian_filter(data,sigma=5) # 0.75 + window 4 less conservative
    fp = findpeaks(method='topology', window=4)
    results = fp.fit(gauss)
    maxima = np.array(results['Xdetect'])
    print(results['peak'])
    print(results['valley'])
    unique, counts = np.unique(maxima, return_counts=True)
    print(dict(zip(unique, counts)))

    print(len(results['peak']))
    #fp.plot_mesh()
    labeled, num_objects = ndimage.label(maxima)
    

    sliced = {i:list(zip(*np.where(labeled==i))) for i in np.unique(labeled) if i}
    xs, ys = [], []
    for coords in sliced.values():
        for y,x in coords:
            xs.append(x)
            ys.append(y)
    return gauss,xs,ys

img = Image.open('data/test_data-14bit.tif')
nframes = range(img.n_frames)[0:1]
fig, axes = plt.subplots(len(nframes), 2, sharex=True, sharey=True)
cycle = 0
for i,frame in tqdm(enumerate(nframes)):
    ax = axes.ravel()
    img.seek(frame)
    data = np.array(img)
    gauss,xs,ys = label_peaks(data)
    
    ax[cycle].imshow(data)                # 0,
    ax[cycle].axis('off')
    cycle+=1
    #ax[0].set_title('Original')
    ax[cycle].imshow(gauss)               # 2,3
    ax[cycle].axis('off')
    ax[cycle].plot(xs,ys, 'r.')
    cycle+=1
    plt.autoscale(False)
    plt.savefig(f'result-{i}.png', bbox_inches = 'tight')

plt.show()
