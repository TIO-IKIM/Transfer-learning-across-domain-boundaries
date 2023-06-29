import numpy as np
import sys
sys.path.append("/Projects/DATL")
from utility import utils as uu
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from itertools import product
from scipy.stats import sem
from scipy import ndimage
from random import shuffle

import pickle, dill
pickle.Pickler = dill.Pickler
from multiprocessing import Process, Lock, Manager
import multiprocessing as mp
mp.set_start_method("fork")
import pebble, concurrent.futures
import time
from copy import deepcopy
import torch, torchvision.transforms.functional as ttf

dl = uu.getFileList("/Projects/Data_Dump/medical_2D/train_wo6", ".npy")

image_ids = []
for file in dl:
    image_ids.append(file.split("/")[-1].split("_")[0])
image_ids = list(set(image_ids))
shuffle(image_ids) # observe average more quickly to help with sanity

def give_sorted(images):
    im1fs = []
    for i, f in enumerate(images):
        fu = f.split("_")
        fd = fu[-1].split(".")
        fn = fd[0].zfill(3)
        f_new = "_".join(fu[:-1]+[".".join([fn, fd[1]])])
        im1fs.append(f_new)
    im1fs = sorted(im1fs)
    im1f = []
    for i, f in enumerate(im1fs):
        fu = f.split("_")
        fd = fu[-1].split(".")
        fn = str(int(fd[0]))
        f_new = "_".join(fu[:-1]+[".".join([fn, fd[1]])])
        im1f.append(f_new)
    return im1f

def mutual_information_2d(x, y, sigma = 1, eps = 1e-6, normalized = False):
    bins = (64, 64)
    x, y = torch.tensor(x).unsqueeze(0).unsqueeze(0), torch.tensor(y).unsqueeze(0).unsqueeze(0)
    x, y = ttf.resize(x, [256, 256]).squeeze(), ttf.resize(y, [256, 256]).squeeze()
    x, y = x.detach().numpy(), y.detach().numpy()
    jh = np.histogram2d(x.flatten(), y.flatten(), bins = bins)[0]

    ndimage.gaussian_filter(jh, sigma = sigma, mode = "constant", output = jh)

    # compute marginal histograms
    jh = jh + eps
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # compute mutual information
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi

class UB2:

    def __init__(self, im1f, im2f, show_progress = False):
        self.im1f, self.im2f = im1f, im2f
        self.show_progress = show_progress

    def upper_bound(self):
        im1fs = give_sorted(self.im1f)
        im2fs = self.im2f
        im1a, im2a = [np.load(f) for f in im1fs], [np.load(f) for f in im2fs]

        # mutual information baseline
        im1_im2_MIs = []
        for i in uu.tqdm_wrapper(range(len(im1fs)), w = self.show_progress, total = min(100, len(im1fs))):
            np.random.seed(int(time.time())+i)
            im1_im2_MIs.append(mutual_information_2d(im1a[i], im2a[np.random.randint(0, len(im2fs))]))
            if i == 0:
                MI_max = mutual_information_2d(im1a[i], im1a[i])
            if i == 100: # is enough for a stable average, and saves time
                break
        MI_min = np.mean(im1_im2_MIs)

        cd = [] # current dataset
        effective_slices = 0
        slices = 0
        shuffle(im1a)

        if len(im1a) > 300: # Huge scans cause quadratic compute requirements, but dont produce different values to normal scans, skip them
            return 0, 0

        for slice_i in uu.tqdm_wrapper(im1a, w = self.show_progress, total = len(im1a)):
            if len(cd) == 0:
                effective_slices += 1
                slices += 1
                cd.append(slice_i)
            else:
                MI_rels = []
                for slice_j in cd:
                    MI_rel_ij = (mutual_information_2d(slice_i, slice_j) - MI_min) / (MI_max - MI_min)
                    MI_rels.append(MI_rel_ij)
                effective_slices += 1 - max(MI_rels) # only need to test correlated slices because the maximally correlated slice has to be among those
                slices += 1
                cd.append(slice_i)
        
        if effective_slices > 0 and effective_slices < slices: # just in case something weird happens with the math
            return effective_slices, slices
        else:
            return 0, 0
    
    def __call__(self):
        return self.upper_bound()
    
noi = len(image_ids)
image_combos = []
for id1 in tqdm(range(noi)):
    id2 = np.random.randint(id1, len(image_ids))
    im1f = [f for f in dl if image_ids[id1] in f]
    im2f = [f for f in dl if image_ids[id2] in f]
    image_combos.append((im1f, im2f))

n_workers = 32
eff_slices = 0
slices = 0
jc = 0
jn = 0

with pebble.ProcessPool(max_workers = n_workers) as PPP:

    # Make jobs and pool
    tasks = []
    for im1f, im2f in image_combos:
        tasks.append(UB2(im1f, im2f, show_progress = False))
    futures = [PPP.schedule(task, timeout = 600) for task in tasks]
    
    # Wait for task completion
    try:
        for future in tqdm(concurrent.futures.as_completed(futures), total = len(image_combos)):
            try:
                e, s = future.result()
                eff_slices += max(0, e) 
                slices += s
                jc += 1
                if jc % 50 == 0:
                    print(f"Jobs finished [{jc}/{len(image_combos)}], S_UB = {slices/eff_slices:.4f}")
            except KeyboardInterrupt:
                raise
            except TimeoutError:
                jc += 1
                jn += 1
                if jc % 50 == 0 and not jc == 0:
                    print(f"Jobs finished [{jc}/{len(image_combos)}], S_UB = {slices/eff_slices:.4f}")
            except:
                jc += 1
                jn += 1
    except KeyboardInterrupt:
        raise
print(f"Jobs finished [{jc}/{len(image_combos)}], S_UB = {slices/eff_slices:.4f}")
print(f"Jobs failed: {jn}")