import numpy as np
import gzip
import os, sys, shutil
sys.path.append("/Projects/DATL")
from tqdm.auto import tqdm
from utility import utils as uu
from random import shuffle, choice

import pickle, dill
pickle.Pickler = dill.Pickler
from multiprocessing import Process, Lock, Manager
import multiprocessing as mp
mp.set_start_method("fork")
import pebble, concurrent.futures

base_dir = "/Projects/Data_Dump/medical_2D/train_all/"

dl = uu.getFileList(base_dir, ".npy")
image_ids = {}
for file in tqdm(dl):
    image_id = file.split("/")[-1].split("_")[0]
    image_ids[str(image_id)] = {}
for file in tqdm(dl):
    l = file.split("/")[:-1]
    image_id = file.split("/")[-1].split("_")[0]
    frame_no = file.split("/")[-1].split("_")[1].split(".")[0]
    l.append(image_id)
    location = "/".join(l)
    image_ids[str(image_id)][str(frame_no)] = location

max_items = len(image_ids)

class CQ:

    def __init__(self, image_ids):

        self.image_ids = image_ids
        self.aifs = 0
        self.rifs = 0

    def get_full_image(self, image_id):
        
        ks = list(self.image_ids[str(image_id)].keys())
        ks = [str(x) for x in sorted([int(y) for y in ks])]
        return np.stack(
            [np.resize(np.load(self.image_ids[image_id][k]+"_"+str(k)+".npy"), (256, 256)) for k in ks], 
            axis = 0
            )

    def get_rand_image(self, length):
        
        return np.stack(
            [np.resize(np.load(choice(dl)), (256, 256)) for f in range(length)],
            axis = 0
            )

    def get_cq(self, image_id):

        real_image = self.get_full_image(image_id)
        rand_image = self.get_rand_image(len(real_image))
    
        # Compress images
        aifn = f"./Complexity_Dump/{image_id}_ai.gz"
        with gzip.open(aifn, "wb") as out:
            out.write(real_image)
        rifn = f"./Complexity_Dump/{image_id}_ri.gz"
        with gzip.open(rifn, "wb") as out:
            out.write(rand_image)

        # Check compressibility
        self.aifs += os.path.getsize(aifn)
        self.rifs += os.path.getsize(rifn)

        # Remove sample files
        os.remove(aifn)
        os.remove(rifn)
        
    def __call__(self):

        for image_id in self.image_ids:
            try:
                self.get_cq(image_id)
            except KeyboardInterrupt:
                raise
            except:
                pass

        return self.rifs, self.aifs
    
n_workers = 128
global_rifs = 0
global_aifs = 0
jc = 0

with pebble.ProcessPool(max_workers = n_workers) as PPP:

    # Make jobs and pool
    tasks = []
    keys = list(image_ids.keys())
    for i in range(len(image_ids)//10):
        ids = {key: image_ids[key] for key in keys[i*10:(i+1)*10]}
        tasks.append(CQ(image_ids = ids))
    futures = [PPP.schedule(task, timeout = 300) for task in tasks]
    
    # Wait for task completion
    try:
        for future in tqdm(concurrent.futures.as_completed(futures), total = len(keys)//10):
            try:
                rifs, aifs = future.result()
                global_rifs += rifs
                global_aifs += aifs
                jc += 1
                if jc % 100 == 0:
                    print(f"Jobs completed [{jc}/{len(image_ids)//10}], c_avg = {global_rifs/global_aifs:.4f}")
            except KeyboardInterrupt:
                raise
            except:
                jc += 1
                if jc % 100 == 0:
                    print(f"Jobs completed [{jc}/{len(image_ids)//10}], c_avg = {global_rifs/global_aifs:.4f}")
    except KeyboardInterrupt:
        raise