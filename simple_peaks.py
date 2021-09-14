from PIL import Image
import numpy as np
from scipy import ndimage
from findpeaks import findpeaks
from tqdm import tqdm

from multiprocessing import Pool
from os import getpid


def label_peaks(data):
    """ 
    This is the main peak labeling function. Uses the findpeaks library to determine significant peaks
    in an image. The algorithm works REALLY well at labeling peaks when phase separated, but does not 
    filter out the diffuse frames at all. 

    Input: raw image data (single frame)
    Output: a list of slices that contain the topological peaks. 
    """

    # Use findpeaks to label the peaks via topological prominence
    fp = findpeaks(method='topology', scale=True, verbose=0)
    results = fp.fit(data)
    #fp.plot_mesh()

    # Create an array of the peaks labeled for numpy interaction
    maxima = np.array(results['Xdetect'])
    #unique, counts = np.unique(maxima, return_counts=True)
    labeled, num_objects = ndimage.label(maxima)

    # Slice the array to get the coordinates of the labeled pixels. 
    sliced = {i:list(zip(*np.where(labeled==i))) for i in np.unique(labeled) if i}
    return sliced

def worker(data):
    gauss = ndimage.filters.gaussian_filter(data,sigma=1)
    slices = label_peaks(gauss)
    return len(slices)

if __name__ == '__main__':
    img = Image.open('data/test.tif')

    #img = Image.open('data/AVG_TC-olaIs39.tif')

   # x2s = []
   # y2s = []
    all_frames = []
    nframes = range(img.n_frames)
    for i,frame in tqdm(enumerate(nframes)):
        img.seek(frame)
        data = np.array(img)
        all_frames.append(data)
        
    pool = Pool(processes = 6)
    counts = pool.map(worker, all_frames)
    results = [(x-counts[0])/counts[0] for x in counts]
    print(results)