import pickle, dill
# Pickle non-global cacher class with dill
pickle.Pickler = dill.Pickler
import torch.multiprocessing
import pebble, concurrent, concurrent.futures
import numpy as np
import PIL.Image
import nibabel as nb
import pydicom
import torch
import torchvision.transforms.functional as ttf
import torch.nn.functional as nnf
import albumentations as alb
import os, sys
import random
import time
from typing import List, Dict, Tuple, Iterable, Any, Callable, Union
from collections import deque
from copy import deepcopy
from tqdm import tqdm
import yaml
import re

class arg(object):
    pass

def filter_kwargs(filter: List[str, ], **kwargs):
    return {key: kwargs[key] for key in list(kwargs.keys()) if not key in filter}

def leq(a: List, b: List):
    if sorted(a) == sorted(b):
        return True
    else:
        return False

def getFileList(path: str, fileType: str = '.png', layer: int = 0, pbar: bool = True):
    """
    Compiles a list of path/to/files by recursively checking the input path a list of all directories
    and sub-directories for existence of a specific string at the end. Much slower than a shallow
    search. Use a shallow search if you know that recursiveness is not necessary.
    
    Inputs:
    
    path - Path-like or string, must point to the directory you want to check for files.
    fileType - String, all files that end with this string are selected.
    
    Outputs:
    
    fileList - A list of all paths/to/files
    """

    if isinstance(fileType, list):
        fileList = []
        for ft in fileType:
            fileList += getFileList(path = path, fileType = ft, layer = 0, pbar = pbar)
        return sorted(fileList)
    
    fileList = []
    if os.path.isfile(path) and path.lower().endswith(fileType.lower()):
        return [path]
    elif os.path.isdir(path):
        if layer == 0:
            for d in tqdm_wrapper(sorted(os.listdir(path)), w = pbar):
                new_path = os.path.join(path, d)
                fileList += getFileList(new_path, fileType, layer = layer+1)
        else:
            for d in sorted(os.listdir(path)):
                new_path = os.path.join(path, d)
                fileList += getFileList(new_path, fileType, layer = layer+1)
    else:
        # Should never be called
        return []

    return sorted(fileList)

def csv_logger(logfile: str, content: Dict, first: bool = False, overwrite: bool = False):
    # Make dirs if not existing yet
    os.makedirs("/".join(logfile.split("/")[:-1]), exist_ok=True)

    if os.path.exists(logfile) and overwrite is False:
        if first is True:
            print("Logfile already exists and overwrite is False. Logging to existing file.")
        with open(logfile, "a+") as c:
            c.write(','.join(str(item) for item in list(content.values()))+"\n")
    elif (os.path.exists(logfile) and overwrite is True) or not os.path.exists(logfile):
        if os.path.exists(logfile) and first is True:
            os.remove(logfile)
        with open(logfile, "a+") as c:
            if first is True:
                c.write(','.join(str(name) for name in list(content.keys()))+"\n")
            c.write(','.join(str(item) for item in list(content.values()))+"\n")
    else:
        pass

def yaml_config_hook(config_file: str):
    """
    Load a yaml config.
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def set_bit(v, index, x):
  mask = 1 << index
  v &= ~mask
  if x:
    v |= mask
  return v

def can_be_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_hex(str: str):
    if any([x in str for x in ["a", "b", "c", "d", "e", "f"]]):
        return True
    else:
        return False

def sh(x: Union[float, int, ]):
    assert isinstance(x, float) or isinstance(x, int)
    ls=["","k","M","B","T","Q"]
    i=0
    if abs(x)<1:
        return "0"
    s="-" if x<0 else ""
    while i!=(len(ls)+1):
        if i==6:
            raise ValueError(f"x is >= 1e{3*len(ls)} and can't be shorthanded. Wow.")
        elif x//(10**(3*i))>=1 and x//(10**(3*(i+1)))==0:
            break
        else:
            i+=1
    if i==0:
        return f"{s}{x}"
    return f"{s}{x/(10**(3*i)):.1f}{ls[i]}"

def truesum(l: Iterable):
    return sum([1 if x is True else 0 for x in l])

def tqdm_wrapper(it: Iterable, w: bool, total = None):
    if w:
        if total is None:
            return tqdm(it)
        else:
            return tqdm(it, total = total)
    else:
        return it

def vtab(s: str, t: int):
    if len(s)>=t:
        return s+" "
    else:
        while len(s)<t:
            s+=" "
        return s

def get_num_params(model: torch.nn.Module, pbar: bool = False):
    """
    Count the named and unnamed parameters of a model.
    If the values in the tuple are not the same, something is probably quite broken in the model.
    """
    names = []
    nums = []
    np = 0
    for n, p in tqdm_wrapper(list(model.named_parameters()), w=pbar):
        names.append(n)
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        np += nn
        nums.append(nn)
    pp = 0
    for p in tqdm_wrapper(list(model.parameters()), w=pbar):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return np, pp

def unravel_model(model: torch.nn.Module, pbar: bool = False):
    """
    Unravel a model, spitting out all named layers in instantiation (not necessarily execution) order.
    Parameter counts are rounded to the nearest factor of 1000, plus one decimal place, if applicable.
    """
    names = []
    nums = []
    np = 0
    for n, p in tqdm_wrapper(list(model.named_parameters()), w=pbar):
        names.append(n)
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        np += nn
        nums.append(nn)
    pp = 0
    for p in tqdm_wrapper(list(model.parameters()), w=pbar):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    nums.append(pp-np)
    names.append("Unnamed params (should be zero)")
    return [sh(num) for num in nums], names

def make_ImageNet_targets(self, size: str = "1k"):
    # If not available in self, load name table and val solutions into dict
    if not hasattr(self, "name_table"):
        bds = self.base_dir.split("/")
        while True:
            if "imagenet" in bds[-1]:
                _ = bds.pop(-1)
                break
            else:
                _ = bds.pop(-1)
        with open(f"{'/'.join([b for b in bds])}/imagenet-1k/LOC_synset_mapping.txt") as o:
            lines = o.readlines()
            codes = [line[:9] for line in lines]
            name_lists = [[name[1:] for name in line[9:].split(",")] for line in lines]
            self.name_table = {code: names for code, names in zip(codes, name_lists)}
        with open(f"{'/'.join([b for b in bds])}/imagenet-1k/LOC_val_solution.csv") as o:
            self.val_solutions = {l.split(",")[0]: l.split(",")[1].split(" ")[0] for i, l in enumerate(o.readlines()) if i != 0}
    
    # If not available in self, load class table into dict
    if not hasattr(self, "code_table"):
        with open(f"{'/'.join([b for b in bds])}/imagenet-1k/LOC_synset_mapping.txt") as o:
            lines = o.readlines()
            codes = [line[:9] for line in lines]
        if "21k" in size:
            raise NotImplementedError
        if "1k" in size:
            self.code_table = {code: i for i, code in enumerate(codes)}
        elif "100" in size:
            self.code_table = {code: i for i, code in enumerate(codes[0::10])}
        elif "10" in size:
            self.code_table = {code: i for i, code in enumerate(codes[0::100])}

def get_ImageNet_target(self, oidx: int, File: str = None):
    if File is None:
        File = self.FileList[oidx]

    # Get target
    if self.state == "train":
        decoded_target = self.code_table[File.split("/")[-2]]
    elif self.state == "val":
        encoded_target = self.val_solutions[File.split("/")[-1].split(".")[0]]
        decoded_target = self.code_table[encoded_target]
    elif self.state == "test":
        # Test solutions are not available publicly for ImageNet-1k!
        # Assume we use val data instead - if not, give back -1 as dummy target class
        try:
            encoded_target = self.val_solutions[File.split("/")[-1].split(".")[0]]
            decoded_target = self.code_table[encoded_target]
        except:
            decoded_target = -1
    else:
        NotImplementedError(f"Unknown dataset state '{self.state}'.")
    return torch.tensor(decoded_target)

def get_ImageNet_target_name(self, oidx: int, File: str = None):
    if File is None:
        File = self.FileList[oidx]

    decoded_name = self.name_table[File.split("/")[-2]]
    return decoded_name

def get_CTBR_target(self, oidx: int, File: str = None):
    classes = {
        0: ["Head", "Skull", "Kopf", "Neck", "Hals", "Schädel", "Schaedel", "Hirn", "Gehirn", "CCT", "Cranial", "Cranium"],
        1: ["Thorax", "Tho", "Thx", "Chest", "Torso"],
        2: ["Abdomen", "Abd", "Oberbauch", "Bauch", "Unterbauch"],
        3: ["Arm", "Ellenbogen", "Elbow", "Hand", "Wrist", "Handgelenk", "Shoulder", "Schulter", "Leg", "Bein", "Knie", "Knee", "Oberschenkel", "Schenkel", "Unterschenkel", "Calf", "Calves", "Upper Calf", "Lower Calf", "Extremity", "Extremities", "Extremität", "Extremitäten"],
        4: ["Hip", "Huefte", "Hüfte", "Pelvis", "Becken"],
        5: ["Spine", "Spinal", "WS", "Wirbelsäule", "Wirbelsaeule"],
        6: ["", "Unknown", "Wholebody"],
        }

    if File is None:
        File = self.FileList[oidx]
    
    tname = get_CTBR_target_name(self=self, oidx=oidx, File=File)
    tid = 0
    while True:
        if any([c.lower() in tname.lower() for c in classes[tid]]):
            break
        if tid == 9:
            break
        tid += 1
    return torch.tensor(tid)

def get_CTBR_target_name(self, oidx: int, File: str = None):
    if File is None:
        File = self.FileList[oidx]

    try:
        file_desig = File.split("/")[-1]
        if is_hex(file_desig):
            origin_dir = "/Projects/RadNet/Rene/BCA-Oncology-Data"
            fdir = (File.split("/")[-1].split("_")[-2])
            dcms = getFileList(origin_dir+"/"+fdir, fileType=".dcm", layer=1)
            assert len(dcms) == 1
            dcm_file = dcms[0]
        else:
            origin_dir = "/Projects/RadNet/DLP/preprocessed"
            file_no = file_desig.split("_")[0]
            fdir = str( int(file_no) // 1024 )
            fnam = str( int(file_no) % 1024  ).zfill(4)
            dcm_file = origin_dir+"/"+fdir+"/"+fnam+".dcm"
        meta = pydicom.dcmread(dcm_file)
        tname = meta[0x0018, 0x0015].value
    except:
        tname = ""
    return tname

def get_LiTS_target(self, oidx: int, File: str = None):
    if File is None:
        File = self.FileList[oidx]
    fd = File.split("/")[-1]
    fdir = File.split("/")[:-2]
    vn, sl = (int(s) for s in re.split('\-|\_|\.', fd) if all([x.isdigit() for x in s]))
    
    # Liver mask
    try:
        with PIL.Image.open("/".join(fdir)+f"/segmentations/segmentation-{vn}_livermask_{sl}.png") as f:
            f = f.convert("L")
            array = np.array(f, dtype = np.uint8)
            liver_mask = torch.tensor(array)
            liver_mask = torch.unsqueeze(liver_mask, 0)
    except OSError:
        print(f"OSError encountered for file {File}")
        liver_mask = torch.zeros(size = self.shape_desired, dtype = torch.uint8)

    # Tumor mask
    try:
        with PIL.Image.open("/".join(fdir)+f"/segmentations/segmentation-{vn}_lesionmask_{sl}.png") as f:
            f = f.convert("L")
            array = np.array(f, dtype = np.uint8)
            tumor_mask = torch.tensor(array)
            tumor_mask = torch.unsqueeze(tumor_mask, 0)
    except OSError:
        print(f"OSError encountered for file {File}")
        tumor_mask = torch.zeros(size = self.shape_desired, dtype = torch.uint8)

    return [liver_mask, tumor_mask]

def get_PVOC_target(self, oidx: int, File: str = None):

    PVOC_colors = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]

    if File is None:
        File = self.FileList[oidx]

    # Find corresponding segmentation file
    dsdir = "/".join(File.split("/")[:-2])
    fname = File.split("/")[-1].split(".")[0]
    segfile = f"{dsdir}/SegmentationClass/{fname}.png" # WHY DO YOU USE JPG IMAGES BUT PNG SEGMENTATIONS AAAAH

    # Load image
    with PIL.Image.open(segfile) as f:
        image = np.array(f.convert("RGB"), dtype = np.uint8)

    # Convert to one-hot, ignoring the 255 background label, and move channel dim
    segmentation_mask = np.zeros((image.shape[0], image.shape[1], len(PVOC_colors)), dtype=np.int64)
    for label_index, label in enumerate(PVOC_colors):
        segmentation_mask[:, :, label_index] = np.all(image == label, axis=-1).astype(np.int64)

    # Convert to CHW tensor
    oh_targets = torch.tensor(segmentation_mask, dtype = torch.long)
    oh_targets = oh_targets[:, :, 1:].moveaxis(-1, 0) # DEBUG, toss out regular background

    return torch.split(oh_targets, 1, dim = 0)

def load_4_channel_MRI_image(self, oidx: int, File: Union[str, None, ] = None, drop_channel: str = "t2"):
    """
    Replacement for _load in the MRI case, as we need to throw out one of the channels,
    so we can reuse the pretrained networks that offer only three channels.
    """

    channel_names = {"t1": True, "t1ce": True, "t2": True, "flair": True}
    if not drop_channel in list(channel_names.keys()):
        raise ValueError("You have to choose one channel to drop, otherwise the network will receive too many input channels!")
    else:
        channel_names[drop_channel] = False

    # Find file
    if File is None:
        File = self.FileList[oidx]

    file_path = "/".join(File.split("/")[:-1])
    file_id = "_".join(File.split("/")[-1].split("_")[:-1])

    # Load
    if self.df == ".npy":
        channels = []
        for channel, use in channel_names.items():
            if use is True:
                current_channel = np.load(f"{file_path}/{file_id}_{channel}.npy")
                channels.append(current_channel)
        array = np.stack(channels, axis = 0)
        target = self._get_target(oidx, File)
        tensor = torch.tensor(array, dtype = (torch.float16 if self.cache_precision == 16 else torch.float32))
    else:
        raise NotImplementedError(f"load_4_channel_MRI_image expects '.npy'-files, not '{self.df}'-files.")

    # Return everything
    return tensor, oidx, target

def get_BraTS_target(self, oidx: int, File: str = None):

    """
    Get BraTS targets. Instead of learning on ET, WT, and TC, we learn on the original
    segmentations: NET+NCR, ED, and NT. The eval metrics later construct ET, WT, and TC, instead.
    """

    # Find file
    if File is None:
        File = self.FileList[oidx]

    file_path = "/".join(File.split("/")[:-1])
    file_id = "_".join(File.split("/")[-1].split("_")[:-1])

    # Load
    tumor_array = np.load(f"{file_path}/{file_id}_seg.npy")
    tumor_mask = torch.tensor(tumor_array, dtype = torch.int64)
    # Class 3 does not exist, so we make class 4 into class 3, then do one_hot
    tumor_mask = torch.clamp(tumor_mask, 0, 3)
    tumor_mask = nnf.one_hot(tumor_mask, num_classes = 4).moveaxis(-1, 0)
    # Recast to uint8 to save memory
    tumor_masks = torch.split(tumor_mask.to(dtype = torch.uint8), 1, dim = 0)

    # Return everything
    return [m for m in tumor_masks[1:]]

def make_CX8_targets(self):

    cnames = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
    self.code_table = {
        cname: i for i, cname in enumerate(cnames)
    }

    with open("./datasets/CX8/Data_Entry_2017.csv", "r") as o:
        lines = o.readlines()
    splits = [line.split(",") for line in lines[1:]]
    self.targets = {split[0]: split[1].split("|") for split in splits}

def get_CX8_target(self, oidx: int, File: str = None):
    
    if File is None:
        File = self.fileList[oidx]
    
    target = sum([2**self.code_table[cname] for cname in self.targets[File.split("/")[-1]]])
    target = torch.tensor(target).to(torch.long)

    return target

def long_to_onehot(l: Union[int, List, torch.LongTensor], c: int = 64):
    """
    Turn a multi-clss long into a one-hot representation by converting it to binary, reversing it, and then converting to a tensor.
    For example, let l = 14 and c = 8, then 14 would convert to 00001110, meaning classes 1, 2, and 3 (if beginnging the count at 0).
    
    If the input is a LongTensor instead of a single Long, the Tensor is returned with an additional dimension appended for the one-hot
    encoded entries. The LongTensor must be 1-D.
    
    If it is a List, a 2-D tensor is returned, where the original list dimension is at dim=0.

    All one-hot "class probabilities" are in dtype torch.float32.
    """
    if isinstance(l, torch.Tensor):
        assert l.dtype == torch.int64
        assert len(l.size()) == 1
        d = l.device
        l = l.tolist()
        bs_l = [str(bin(le)[2:].zfill(c)) for le in l]
        vec_l = [torch.tensor([int(n) for n in list(bs)[::-1]]) for bs in bs_l]
        vec = torch.vstack(tensors = vec_l)
        vec = vec.to(device = d, dtype = torch.float32)
    elif isinstance(l, list):
        if isinstance(l[0], torch.Tensor):
            d = l[0].device
            bs_l = [str(bin(le)[2:].zfill(c)) for le in l]
            vec_l = [torch.tensor([int(n) for n in list(bs)[::-1]]).to(device = d, dtype = torch.float32) for bs in bs_l]
        else:
            bs_l = [str(bin(le)[2:].zfill(c)) for le in l]
            vec_l = [torch.tensor([int(n) for n in list(bs)[::-1]], dtype = torch.float32) for bs in bs_l]
        vec = torch.vstack(vec_l)
    else:
        bs = str(bin(l)[2:].zfill(c))
        vec = torch.tensor([int(n) for n in list(bs)[::-1]])
        vec = vec.to(torch.float32)
    return vec

def convert_tensor_to_opencv_array(tensor: torch.Tensor, as_type = None):
    """
    Convert a tensor from CHW tensor to HWC numpy array. Number of leading dimensions can be arbitrary.
    If the tensor is already in HWC format, no operation is performed except casting to the array with as_type.
    If the tensor has no channel dimension, a dimension will be added at dim = -3.
    If it is unclear what the channel dimension is, raise an error.
    """
    s = tensor.size()
    if len(s) == 2:
        tensor = tensor.unsqueeze(-3)
        s = tensor.size()
    if s[-1] not in [1, 3]: # Is not HWC
        if s[-3] in [1, 3]: # Is CHW
            array = tensor.movedim([-2, -1, -3], [-3, -2, -1]).detach().cpu().numpy()
        else: # Is unknown format
            raise ExclusionError(f"Tensor has unusable channel dimensions at dim=-3/-1: {s[-3]}/{s[-1]}.")
    if as_type is not None:
        array = array.astype(as_type)
    return array

def convert_opencv_array_to_tensor(array: np.ndarray, as_type = None):
    """
    Convert an array from HWC numpy array to CHW tensor. Number of leading dimensions can be arbitrary.
    If the array is already in HWC format, no operation is performed except casting to the tensor with as_type.
    If the array is missing a channel dimension, one is unsqueezed at dim = -1.
    If it is unclear what the channel dimension is, raise an error.
    """
    s = array.shape
    if len(s) == 2:
        array = array.expand_dims(-1)
        s = array.shape
    if s[-3] not in [1, 3]: # Is not CHW
        if s[-1] in [1, 3]: # Is HWC
            tensor = torch.tensor(np.moveaxis(array, [-1, -3, -2], [-3, -2, -1]))
        else: # Is unknown format
            raise ExclusionError(f"Array has unusable channel dimensions at dim=-3/-1: {s[-3]}/{s[-1]}.")
    if as_type is not None:
        tensor = tensor.to(dtype = as_type)
    return tensor

def save_model(model, path: str):
    """
    Save the model to path.
    """

    os.makedirs(path.split("/")[0], exist_ok = True)
    
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

def ltup(l: List):
    """
    Turn an arbitrarily nested list into a tuple of the same structure.
    """
    if isinstance(l, list):
        return tuple(ltup(sl) for sl in l)
    else:
        return l

def itemize_and_cache(data, idx, targets, cache_ds):
    cd = data.clone().detach().numpy()
    ct = targets.clone().detach().numpy()
    cd = np.squeeze(np.split(cd, len(cd), axis=0), axis=1)
    ct = np.squeeze(np.split(ct, len(ct), axis=0), axis=1)
    ci = idx.clone().detach().tolist()
    cache_ds._batch_to_cache(cd, ct, ci)

class ExclusionError(Exception):
    pass

class CacheError(Exception):
    pass

class Custom_Dataset(torch.utils.data.Dataset):
    
    """
    Custom dataset class for PyTorch, inherits from the PyTorch dataset. Implements the usual
    __init__, __len__, and __getitem__ functions.

    This dataset supports saving all the data you want to train with in a single object.
    This is done by saving every filename in a huge dictionary named FileList and implementing
    subsets which contain indices from this dictionary. Subsets can be split into new subsets
    and merged together. When __len__ is called, the length of the currently used subset is
    returned, unless a subset is specified.

    This dataset implements caching. If opmode is set to "live_cache", the dataset will expect
    a manager kwarg which should be a valid python multiprocessing manager. This manager object
    will receive all data that the dataset has not loaded before (it checks the cache to know this).
    Starting with the second epoch, the datasets will query the cache manager for the data instead
    of loading it from disk. Usually this is a good deal faster than loading from disk.

    If __getitem__ can't find a file on disk or in the cache, or if the file is broken or unusable
    for some reason, it will spit out the error and try to replace it with a random other image.

    Expected input for the dataset includes:
    base_dir - A directory to search in for data if no train/val/test folders are given, usually irrelevant.
    cpu_transforms - Deterministic transforms which will always be performed
    gpu_transforms - Non-deterministic transforms like augmentations, which are usually outsourced to the GPU.
    verbose - Verbosity.
    debug - Always test for nans/infs, raise any errors instead of swallowing them. Slow.
    
    One of tvt, tvt_by_name, tvt_by_filename - tvt stands for Train/Val/Test and this is how you
    usually give the dataset its files.
    tvt - A dictionary containing subset names as keys, and folders as values. The dataset will recursively
     search folders for any files ending on self.df (defaults to '.npy').
    tvt_by_name - A dictionary containing subset names as keys and a string as value. The dataset will
     recursively search self.base_dir for any files containing that string and ending on self.df (defaults
     to '.npy')
    tvt_by_filename - A dictionary containing subset names as keys and lists of files as values. The
     dataset will simply memorize all the filenames as part of the subsets, in the order they were provided.

    Finally, this dataset calls a _get_target function under the hood. This function is specific to
    the data we use, and gets bound to the dataset instance at runtime (meaning it gets access to self
    despite being a function that lives outside the dataset).
    """

    def __init__(self,
                base_dir: str = "./Downloads",
                cpu_transforms: Union[Callable, None, ] = None,
                gpu_transforms: Union[Callable, None, ] = None,
                verbose: bool = False,
                debug: bool = False,
                **kwargs):

        # Set defaults from args and kwargs
        self.base_dir = base_dir
        self.cpu_transforms = cpu_transforms # Deterministic and inexpensive, gets cached and is therefore applied ONCE per datapoint when caching
        self.gpu_transforms = gpu_transforms # Non-deterministic or very expensive, does NOT get cached
        self.verbose = verbose
        self.debug = debug
        self.kwargs = kwargs
        
        # Desired shape before going into transformations
        self.shape_desired = self.kwargs.get("shape_desired", None)

        # Filetype of images
        if isinstance(self.kwargs.get("data_format", ".npy"), list):
            self.df = [ft for ft in self.kwargs.get("data_format", ".npy")]
        else:
            self.df = self.kwargs.get("data_format", ".npy").lower()

        # Randomizer seed
        torch.manual_seed(kwargs.get("seed", 42))
        np.random.seed(kwargs.get("seed", 42))
        random.seed(kwargs.get("seed"), 42)

        # Number of masks
        # classification = 0
        # segmentation = 2+ depending on number of masks 
        # background counts, but is automatically built during loss forward (more efficient)
        self.num_masks = self.kwargs.get("num_masks", 0)

        # Whether to test for nans and infs
        self.test_nan_inf = self.kwargs.get("test_nan_inf", False)

        # Force all color images to be grayscaled (will keep 3 channels)
        self.grayscale_only = self.kwargs.get("grayscale_only", False)

        # How to normalize the images
        self.normalizer = self.kwargs.get("normalizer", "01_clipped_CT")
        # If "windowed" is chosen, a normalization range must be given
        # Normalization ranges compress [X, Y] -> [A, B]. Default for A, B is 0, 1.
        # If X, Y are given, clip input at X, Y. If they are None, no clipping is done.
        self.normalizer_range = self.kwargs.get("normalizer_range", (None, None, 0, 1))

        # Execute Transforms on gpu (faster)? or cpu (during dataloading, slower but sometimes necessary)?
        self.tf_device = self.kwargs.get("tf_device", "cuda")
        if self.tf_device == "gpu" and self.num_masks != 0:
            raise ValueError("Albumentations uses opencv arrays in the transformations. A numpy array cannot be pushed to the GPU. Please set tf_device to 'cpu'.")
        
        # What load mode to operate in (pre_cache, live_cache or disk)
        self.opmode = self.kwargs.get("opmode", "disk")

        # What data precision to cache in, expressed in bit length (16 or 32)
        # To work on the CPU, the precision of the data must be 32bit floating point,
        # as transformations are not implemented for HalfTensor on the CPU.
        self.cache_precision = self.kwargs.get("cache_precision", 16)

        # Start off by indexing the dataset, and add the "base" subset, containing all data
        # If "tvt" is not None, construct the subset, construct the base set from the subsets
        # If "tvt_by_name" is not None, 
        # Subsets can be overlapping or unique
        self.tvt = self.kwargs.get("tvt", None)
        self.tvt_by_name = self.kwargs.get("tvt_by_name", None)
        self.tvt_by_filename = self.kwargs.get("tvt_by_filename", None)
        if sum([1 if specified is not None else 0 for specified in [self.tvt, self.tvt_by_name, self.tvt_by_filename]]) > 1:
            raise ValueError("Please only specify either a train-val-test split with the tvt option, or a tvt-split-by-(file)name with the tvt_by_(file)name option, but not more than one.")
        if self.tvt is None and self.tvt_by_name is None and self.tvt_by_filename is None:
            _ = self._index_full_dataset()
            self.subsets = {"base": [x for x in range(len(self.FileList))]}
            self.caching_complete = {"base": False}
        elif self.tvt is not None:
            if self.verbose:
                print("Warning: tvt option does not guarantee unique train, val and test datasets if either directory contains a symlink to another directory!")
            self.FileList = {}
            self.subsets = {}
            self.caching_complete = {"base": False}
            for key in self.tvt:
                sublist = getFileList(self.tvt[key], fileType = self.df, pbar = self.verbose)
                self.subsets[key] = [len(self.FileList)+x for x in range(len(sublist))]
                fll = len(self.FileList)
                subdict = {idx+fll: filename for idx, filename in enumerate(sublist)}
                self.FileList.update(subdict)
                self.caching_complete[key] = False
            self.subsets["base"] = [x for x in range(len(self.FileList))]
        elif self.tvt_by_name is not None:
            if self.verbose:
                print("Warning: tvt_by_name option does not guarantee unique train, val and test datasets if matched strings arenot unique to the dataset!")
            self.FileList = {}
            self.subsets = {}
            self.caching_complete = {"base": False}
            for sname, fstr in self.tvt_by_name.items():
                sublist = [filename for filename in tqdm(getFileList(self.base_dir, fileType = self.df, pbar = self.verbose)) if fstr in filename]
                self.subsets[sname] = [len(self.FileList)+x for x in range(len(sublist))]
                fll = len(self.FileList)
                subdict = {idx+fll: filename for idx, filename in enumerate(sublist)}
                self.FileList.update(subdict)
                self.caching_complete[sname] = False
            self.subsets["base"] = [x for x in range(len(self.FileList))]
        elif self.tvt_by_filename is not None:
            if self.verbose:
                print("Warning: tvt_by_filename option does not guarantee unique train, val and test datasets if filenames are not unique!")
            self.FileList = {}
            self.subsets = {}
            self.caching_complete = {"base": False}
            for sname, sfiles in self.tvt_by_filename.items():
                sublist = [filename for filename in tqdm(sfiles)]
                self.subsets[sname] = [len(self.FileList)+x for x in range(len(sublist))]
                fll = len(self.FileList)
                subdict = {idx+fll: filename for idx, filename in enumerate(sublist)}
                self.FileList.update(subdict)
                self.caching_complete[sname] = False
            self.subsets["base"] = [x for x in range(len(self.FileList))]

        # Select a specific subset (can be changed from outside later on) and set training state
        self.sss = "base"
        self.state = "train"
        if self.verbose:
            for sname in self.subsets:
                print(f"Subset {sname}: {len(self.subsets[sname])} datapoints.")

        # If desired, cache the entire dataset to memory at the start
        # If your data does not fit into memory, this will fail
        if "cache" in self.opmode:
            if self.shape_desired != None:
                # Assume torch.float16 image, 3 input channels
                image_size = 2*3*self.shape_desired[0]*self.shape_desired[1]*len(self)//(1024**2)
                if self.num_masks == 0:
                    # Assume max 10 bytes per tensor
                    targets_size = 1*10*len(self)//(1024**2)
                else:
                    # Assume 1 unsigned 8-bit ByteTensor per masks, 3 input channels
                    targets_size = self.num_masks * 1*3*self.shape_desired[0]*self.shape_desired[1]*len(self)//(1024**2)
                print(f"Maximum cache size: {image_size + targets_size} MiB (per worker with non-shared access).")
            else:
                print(f"Warning: Maximum cache size cannot be determined if shape_desired is None.")


        self.manager = self.kwargs.get("manager", None)
        if self.manager is not None:
            torch.multiprocessing.set_sharing_strategy("file_system")
            self.cachedFileList = self.manager.shared_cache
        else:
            self.cachedFileList = {idx: None for idx, File in enumerate(self.FileList)}

        print("Dataset init complete.")

    def __len__(self, subset: str = None):
        # Return the length of the currently used subset, 
        # unless subset is specified, in which case it returns the subset's length
        if subset is None:
            return len(self.subsets[self.sss])
        elif subset in list(self.subsets.keys()):
            try:
                return len(self.subsets[subset])
            except:
                raise KeyError(f"Subset {subset} does not exist.")
        else:
            print(subset, list(self.subsets.keys()))
            raise RuntimeError("len unavailable for some reason.")
        
    def _index_full_dataset(self):
        # Find all data
        print("Indexing dataset. This may take a moment ...")
        self.FileList = {idx: filename for idx, filename in enumerate(getFileList(self.base_dir, fileType=self.df))}
        print("Dataset indexed. Dataset length:", str(len(self.FileList)))
        return None

    def _cache_dataset_zeroW(self, name: str = None):

        """
        Cache the current selected subset, or a specific one, if given a name.
        Must be called manually, is not called in init (because the task-specific
        get_target is not yet a method of the class at that time).
        """

        if name is not None:
            prev_name = self.sss
            self.set_used_subset(name)

        print("Caching dataset. This may take a moment ...")

        for idx, oidx in tqdm(enumerate(self.subsets[self.sss]), total=len(self.subsets[self.sss])):
            try:
                tensor, _, target = self._load(oidx = oidx, File = self.FileList[oidx])
                tensor, target = self._ensure_conformity(tensor, target)
                tensor, target = self.apply_cpu_transforms(tensor, target)
                tensor = self._ensure_safety(tensor)
                target = self._ensure_safety(target)
                self._load_to_cache((tensor, target), oidx)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if self.verbose:
                    print(repr(e))
                if self.debug:
                    raise
                self._load_to_cache((None, None), oidx)

        self.set_caching_complete(self.sss)
        if self.sss == "base":
             for sname in self.subsets:
                self.set_caching_complete(sname)

        self.set_used_subset(prev_name)

    def _cache_full_dataset(self):
        # Load dataset into memory (makes no check for available memory)
        print("Caching dataset. This may take a moment ...")
        raise NotImplementedError("Broken.")

        # Prevent torch from leaking FDs
        torch.multiprocessing.set_sharing_strategy("file_system")

        '''clone = arg()
        for var in vars(self):
            if var not in ["FileList", "cachedFileList", "subsets"]:
                setattr(clone, var, getattr(self, var))'''

        # Outsource the loading processes to a worker pool
        cpu_count = max(torch.multiprocessing.cpu_count()//4, 1)
        with pebble.ProcessPool(max_workers = min(cpu_count, 8)) as PPP:
            cs = 1024
            SubLists = [self.FileList[(x*cs):min((x+1)*cs, len(self.FileList))] for x in range((len(self.FileList)//cs)+1)]
            tasks = [cacheloader(oidxs = [n*cs+x for x in range(cs)], 
                                 Files = SubList,
                                 ds = self) 
                    for n, SubList in enumerate(SubLists)]
            futures = [PPP.schedule(task) for task in tasks]
            try:
                for future in tqdm(concurrent.futures.as_completed(futures), total=(len(self.FileList)//cs +1)):
                    try:
                        results = future.result()
                        for result, oidx in results:
                            self._load_to_cache(result, oidx)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(repr(e))
            except KeyboardInterrupt:
                # If interrupted, return all resources immediately, by gracefully murdering any jobs
                # which would block the Pool's shutdown
                print("Canceling all jobs and exiting. Please wait. This may take a couple of seconds.")
                print("If any additional jobs start running despite cancellation, they were already in the buffer.")
                print("This is normal. These jobs will be cancelled, too, once they start running.")
                if self.verbose:
                    print("Canceling pending futures.")
                for future in futures:
                    # Cancel regularly if it has not started yet
                    if future._state == "PENDING":
                        _ = future.cancel()
                # Cancel remaining futures
                if self.verbose:
                    print("Canceling running futures.")
                for future in futures:
                    if future._state == "RUNNING":
                        _ = future.cancel()
                # Cancel futures coming out of the pipe
                while True:
                    time.sleep(2)
                    if not any(future._state == "RUNNING" for future in futures):
                        break
                    else:
                        for future in futures:
                            if future._state == "RUNNING":
                                _ = future.cancel()
                if self.verbose:
                    print("All futures successfully cancelled.")
        self.set_caching_complete()
        print("Dataset cached into RAM.")
        return None

    def set_caching_complete(self, name: str = None, check_count: bool = False):
        if name is None:
            for key in self.caching_complete:
                self.caching_complete[key] = True
            if self.verbose:
                print(f"caching_complete set to 'True' for all subsets.")
        else:
            self.caching_complete[name] = True
            if self.verbose:
                print(f"caching_complete set to 'True' for subset '{name}'.")
        if self.verbose and check_count:
            if name is None:
                name = self.sss
            cached = len([1 for i, _ in tqdm(enumerate(self.subsets[name])) if self._load_from_cache(self.subsets[name][i]) is not None])
            cachelen = len(self.subsets[name])
            print(f"[DEBUG] cache content: [{cached}/{cachelen}], ~{((cached/cachelen)*100):.1f}%")

    def split_dataset(
        self, 
        target: str, 
        names: List[str, ], 
        fracs: List[float, ] = None, 
        nums: List[int, ] = None, 
        shuffle: bool = True):
        # Adds subsets to the dataset, extending the dictionary where the member indices are kept
        # The default dataset is called "training"

        if fracs is None and nums is None:
            raise ValueError("Must supply one of fracs, nums.")
        if fracs and nums:
            raise ValueError("Cannot select dataset length by fraction and exact length. Choose one.")
        if names and fracs and not len(names)==len(fracs) or (names and nums and not len(names)==len(nums)):
            raise IndexError("Names and fracs or names and nums must be of the same length.")
        if names and not all(isinstance(n, str) for n in names):
            raise TypeError("All n in names must be of type string.")
        if fracs and sum(fracs) > 1. or nums and sum(nums) > len(self.subsets[target]):
            raise ValueError("Cannot split dataset. Sum of subsets must be smaller than target dataset")
        if nums and not all(isinstance(n, int) for n in nums):
            raise TypeError("All n in nums must be integers.")
        if shuffle is True:
            indices = deepcopy(self.subsets[target])
            random.shuffle(indices)
        else:
            indices = deepcopy(self.subsets[target])

        if fracs:
            # You can lose datapoints due to rounding (because of int acting like floor)
            nums = [int(frac*len(indices)) for frac in fracs]

        total = 0
        for i, name in enumerate(names):
            self.subsets[name] = indices[total:total+nums[i]]
            total += nums[i]

        del(indices)

        # Set cache state for new items
        if self.caching_complete[target] is True:
            for name in names:
                self.caching_complete[name] = True
        else:
            for name in names:
                self.caching_complete[name] = False

        return None

    def create_subset(self, target: str, name: str, specific_indices: Union[List, None, ] = None, specific_files: Union[List, None, ] = None):
        """
        Create a subset from a subset. To create a subset from the entire dataset, specify 'base' as target.
        when giving specific indices, a list of ints is accepted. These are expected to be valid indices of the target subset.
        When giving specific files, a list of strings is accepted. If any of the file names in the subset contain any of the strings in specific_files, these files will be included in the new subset. Warning: This is slow.
        """

        if specific_indices is None and specific_files is None:
            raise ValueError("Must provide one of specific_indices or specific_files.")
        if specific_indices is not None and specific_files is not None:
            raise ValueError("Must provide either specific_indices or specific_files.")

        if specific_indices is not None:
            if all(isinstance(si, int) for si in specific_indices):
                raise TypeError("specific_indices must be a list of int, containing integers (that are valid indices of the target dataset.")
            collection = self.subsets[target][specific_indices]

        if specific_files is not None:
            if not all(isinstance(si, str) for si in specific_files):
                raise TypeError("specific_files must be a list of str, containing specific filenames (that are in FileList).")
            file_names = [self.FileList[i] for i in [self.subsets["base"][j] for j in self.subsets[target]]]
            subset_ids = [x for x in self.subsets[target]]
            ids = self.subsets["base"][subset_ids]
            collection = [id for id, file_name in tqdm_wrapper(zip(ids, file_names), w=self.verbose) if any(string in file_name for string in specific_files)]

        self.subsets[name] = collection
        if self.caching_complete[target] is True:
            self.caching_complete[name] = True
        else:
            self.caching_complete[name] = False

    def merge_datasets(self, name: str, targets: Tuple[str]):
        collection = []
        for t in targets:
            collection.extend(self.subsets[t])
        self.subsets[name] = collection

        # Set cache state
        if all([self.caching_complete[target] for target in targets]):
            self.caching_complete[name] = True
        else:
            self.caching_complete[name] = False
        return None

    def n_fold_CV_split(self, target: str, n_folds: int, fracs: List[float, ] = [0.8,0.1,0.1], names: List[str, ] = ["train", "val", "test"]):

        # Check inputs
        if sum(fracs)!=1:
            raise ValueError(f"Sum of splits in n-fold cross validation must be 1 but is {sum(fracs)}")
        if len(fracs) not in [2, 3]:
            raise NotImplementedError(f"Cross validation requires making 2 or 3 splits, but was asked for {len(fracs)}")
        if len(fracs) != len(names):
            raise ValueError(f"fracs and names must have same length, but are {len(fracs)} and {len(names)}.")

        # Shuffle the list only once and make a subset from it.
        self.subsets["temporary"] = deepcopy(self.subsets[target])
        random.shuffle(self.subsets["temporary"])

        # Make folds
        for n in range(n_folds):
            self.split_dataset(target = "temporary",
                                names = ["{}_f{}_{}".format(target, str(n), name) for name in names],
                                fracs = fracs,
                                shuffle = False)
            # You can lose datapoints due to rounding (because of int acting like floor)
            self.subsets["temporary"] = deque(self.subsets["temporary"]).rotate(int(max(fracs)*len(self.subsets["temporary"])))
        del(self.subsets["temporary"])

        return None

    def set_used_subset(self, name: str):
        # Set the current subset which __getitem__ will exclusively select from.
        if name is None or name == "base":
            self.sss = "base"
        elif name in list(self.subsets.keys()):
            self.sss = name
        else:
            raise KeyError(f"Subset '{name}' does not exist.")
        if self.verbose:
            print(f"Now using subset '{self.sss}'.")
        return None

    def set_state(self, state: str):
        # Set dataset state to training/testing/whatever.
        self.state = state
        if self.verbose:
            print(f"Dataset now in state '{self.state}'.")
        return None

    def _get_target(self, oidx: int, File: Union[str, None, ] = None):
        # Loading function that generates the target(s), does not do safeties.
        # Any replacement must take self, oidx, File, use self.df and return target.
        # This function is just a dummy and is never used.
        if File is None:
            File = self.FileList[oidx]

        return torch.tensor(0)

    def _load(self, oidx, File: Union[str, None, ] = None):
        """
        Main loading function. Loads from self.FileList.
        Any replacement must take self, oidx, File, use self.df, and return (tensor, idx, target).
        """

        # Find file
        if File is None:
            File = self.FileList[oidx]
        filetype = "."+File.split(".")[-1].lower()

        # Safety check
        allowed = [".npy", ".npz", ".pth", ".jpeg", ".jpg", ".png", ".nii", ".nii.gz"]
        if isinstance(self.df, list):
            for ft in self.df:
                if not ft.lower() in allowed:
                    raise NotImplementedError(f"_load does not support data type '{self.df}'.")
        else:
            if not self.df.lower() in allowed:
                raise NotImplementedError(f"_load does not support data type '{self.df}'.")

        # Read the image
        if filetype == ".npy":
            array = np.load(File)
            target = self._get_target(oidx, File)
            tensor = torch.tensor(array, dtype = (torch.float16 if self.cache_precision == 16 else torch.float32))
        elif filetype == ".npz":
            with np.load(File) as loaded:
                array = loaded["image_array"]
                target = loaded["class_array"][0]
                tensor = torch.tensor(array, dtype=(torch.float16 if self.cache_precision == 16 else torch.float32))
        elif filetype == ".pth":
            tensor, target = torch.load(File)
        elif filetype in [".jpeg", ".jpg", ".png"]:
            with PIL.Image.open(File) as f:
                if self.grayscale_only is True:
                    # Convert from any channel number to grayscale - sometimes necessary to spot 4-channel images
                    f = f.convert("L")
                array = np.array(f)
            target = self._get_target(oidx, File)
            tensor = torch.tensor(array, dtype = (torch.float16 if self.cache_precision == 16 else torch.float32))
        elif filetype in [".nii", ".nii.gz"]:
            array = nb.load(File).get_fdata()
            target = self._get_target(oidx, File)
            tensor = torch.tensor(array, dtype = (torch.float16 if self.cache_precision == 16 else torch.float32))
        else:
            raise NotImplementedError(f"_load is not implemented for data of filetype {filetype}.")

        # Return everything
        return tensor, oidx, target

    def _load_from_cache(self, oidx: int):

        """
        Load item from cache. Infer types from numpy types.
        If targets are masks, return list of individual masks for albumentations.
        """

        try:
            item = self.cachedFileList[oidx]
            if isinstance(item[1], list):
                return (torch.as_tensor(item[0]), [torch.as_tensor(mask) for mask in item[1]])
            return (torch.as_tensor(item[0]), torch.as_tensor(item[1]))
        except (KeyError, TypeError): # (does not exist, exists but is empty)
            return None

    def _load_to_cache(self, item: Tuple, oidx: int):

        """
        Load items to cache. If tensor, push to numpy to avoid weird tensor shm errors.
        If targets are masks, convert each item.
        """

        if item[0] is None or item[1] is None:
            item = None
        else:
            if isinstance(item[0], torch.Tensor): 
                d = item[0].cpu().numpy()
            elif isinstance(item[0], np.ndarray):
                d = item[0]
            else:
                raise TypeError(f"Trying to cache image type {type(item[0])}, which is illegal.")
            if isinstance(item[1], torch.Tensor):
                t = item[1].cpu().numpy()
            elif isinstance(item[1], np.ndarray):
                t = item[1]
            elif isinstance(item[1], list) or isinstance(item[1], tuple):
                t = []
                for i, mask in enumerate(item[1]):
                    if isinstance(mask, torch.Tensor):
                        t.append(mask.cpu().numpy())
                    elif isinstance(mask, np.ndarray):
                        t.append(mask)
                    else:
                        raise TypeError(f"Trying to cache target mask (index {i}) type {type(mask)}, which is illegal.")
            else:
                raise TypeError(f"Trying to cache target type {type(item[1])}, which is illegal.")
            item = (d, t)
        self.cachedFileList[oidx] = item

    def _careful_load(self, idx: int):
        # Wraps _load. Contains safeguards against corrupt files, NaNs and Infs. Replaces items where loading caused
        # an exception. If too many loads fail in a row (hardcoded to 10), reraises the error instead of replacing items.

        # Convert index on subset to index on FileList/cachedFileList
        oidx = self.subsets[self.sss][idx]

        try:
            # Load a datapoint. 
            # If caching is enabled and the cache is fully loaded, load from cache, else load from disk
            if self.opmode == "disk":
                tensor, _, target = self._load(oidx)
            else:
                cached_item = self._load_from_cache(oidx)
                if cached_item is not None:
                    tensor, target = cached_item
                elif self.opmode == "live_cache" and self.caching_complete[self.sss] is False:
                    tensor, _, target = self._load(oidx)
                else:
                    raise CacheError(f"Datapoint of oidx {str(oidx)} not found in cache")
            
            
            # Ensure conformity and safety (ensured already for cached items)
            if self.opmode == "disk" or (self.opmode == "live_cache" and self.caching_complete[self.sss] is False):
                tensor, target = self._ensure_conformity(tensor, target)
                tensor, target = self.apply_cpu_transforms(tensor, target)
                
            # Return (keeping a copy in cache, if desired. Surprisingly, this is quite fast despite IPC overhead)
            if self.opmode == "live_cache" and self.caching_complete[self.sss] is False:
                self._load_to_cache((tensor, target), oidx)
                #pass

            return tensor, idx, target

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if self.verbose:
                print("Image "+str(idx)+" at location "+str(self.FileList[oidx])+" could not be loaded. Reason: "+str(repr(e)))
            raise

    def _extend_channel_dim(self, tensor: torch.Tensor, n_channels: int):
        tensor = torch.cat([tensor for x in range(n_channels)], -3)
        return tensor

    def _apply_normalizer(self, tensor: torch.Tensor):
        supported = ["naive", "01_clipped_CT", "imagenet", None]
        if self.normalizer not in supported:
            raise NotImplementedError(f"Normalizer must be one of {supported} but got '{self.normalizer}'.")

        if self.normalizer == "naive":
            tensor -= torch.min(tensor)
            tensor /= torch.max(tensor)

        if self.normalizer == "01_clipped_CT":
            tensor = torch.clip(tensor, min=-1024., max=3072.)
            tensor -= torch.min(tensor)
            tensor /= torch.max(tensor)

        if self.normalizer == "imagenet":
            # Normalize images to 0, 1
            # tensor -= 0.
            tensor /= 255.
            # Apply imagenet values
            imn_normalize = ttf.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensor = imn_normalize(tensor)

        if self.normalizer == "windowed":
            if len(self.normalizer_range) != 4:
                raise ValueError("Normalizer range must be 4-tuple, instead found length "+str(len(self.normalizer_range)))
            if X >= Y or A >= B:
                raise ValueError("The window [X, Y] -> [A, B] for normalization must fulfill X < Y and A < B.")
            X, Y, A, B = self.normalizer_range
            if X is not None or Y is not None:
                tensor = torch.clip(tensor, min = X, max = Y)
            tensor -= torch.min(tensor)
            tensor /= torch.max(tensor)
            tensor *= (B-A)
            tensor += (A)

        if self.normalizer == None:
            pass

        return tensor

    def _apply_resize(self, tensor: torch.Tensor, interpolation: str = "bilinear"):
        # If we receive more than one tensor (e.g. multiple masks), execute on each tensor and return
        if isinstance(tensor, tuple):
            return tuple(self._apply_resize(tensor = t, interpolation = interpolation) for t in tensor)
        elif isinstance(tensor, list):
            return [self._apply_resize(tensor = t, interpolation = interpolation) for t in tensor]
        # We don't care about the shape
        if self.shape_desired is None:
            return tensor
        # Weird shapes
        if self.shape_desired and len(self.shape_desired) != 2:
            raise ValueError("Desired shape must be 2 dimensional.")
        # Shape matches
        if tensor.size()[-2:] == self.shape_desired:
            return tensor
        # Shape does not match
        else:
            imodes = {
                "bicubic": ttf.InterpolationMode.BICUBIC,
                "bilinear": ttf.InterpolationMode.BILINEAR,
                "nearest": ttf.InterpolationMode.NEAREST
            }
            tensor = ttf.resize(tensor, size = self.shape_desired, interpolation = imodes[interpolation], antialias = False)
            return tensor
    
    def _ensure_conformity(self, tensor: torch.Tensor, target: Union[torch.Tensor, List[torch.Tensor, ], ], non_CHW_allowed: bool = False):
        
        """
        Ensures conformity to C x H x W tensor layout.
        If non_CHW_allowed is True, ensures conformity to ... x C x H x W tensor layout, ignoring leading dimensions. If False, the tensor is squeezed, and discarded if that does not reduce it to 3D.
        'non_CHW_allowed' = True is insufficiently tested, use at your own risk.
        If self.grayscale_only is True, convert from assumed RGB to single channel.
        """
        s = tensor.size()
        if len(s) == 2:
            # Add channel dimension, if necessary
            tensor = torch.unsqueeze(tensor, 0)
            tensor = self._extend_channel_dim(tensor, 3)
            s = tensor.size()
        elif len(s) > 3 and non_CHW_allowed is False:
            # Remove extra dimensions, if easily done
            if len(tensor.squeeze().size()) > 3:
                raise ExclusionError(f"Image has at least one dimension (like batch dimension) more than it should. Dimensions of the image tensor were {s}.")
            else:
                tensor = tensor.squeeze()
                s = tensor.size()
        # Tensor should now be ...CHW (or CHW)
        if s[-3] not in [1, 3]:
            if s[-1] in [1, 3]:
                # This most likely means current order is X, Y, C
                # Reverse X, Y, C to C, X, Y if necessary
                tensor = torch.movedim(tensor, [-1, -3, -2], [-3, -2, -1])
                # Our masks do not come with a channel dimension, so no need to change anything there.
                s = tensor.size()
            else:
                # This means something went wrong
                raise ExclusionError(f"Image has unsupported channel dimension. Full image dimensions are: {s}.")
        if s[-3] == 1:
            tensor = self._extend_channel_dim(tensor, 3)
        elif s[-3] == 3 and self.grayscale_only == True:
            tensor = ttf.rgb_to_grayscale(tensor, num_output_channels=3)
        else:
            pass

        # Target and Image dimensions must match for all targets, assuming they are segmentation masks
        if self.num_masks != 0:
            if isinstance(target, list) or isinstance(target, tuple):
                if not all([tensor.size()[-2:] == target[i].size()[-2:] for i in range(len(target))]):
                    raise ExclusionError(f"Image and target dimension do not match for all targets! Image dimension is {tensor.size()}, target dimensions are {[t.size() for t in target]}.")
            elif isinstance(target, torch.Tensor) and not tensor.size()[-2:] == target.size()[-2:]:
                raise ExclusionError(f"Image and target dimensions do not match! Image dimension is {tensor.size()}, target dimension is {target.size()}.")
            else:
                raise ExclusionError(f"Target type ({type(target)}) is not accepted.")
            
        return tensor, target

    def _ensure_safety(self, tensor: torch.Tensor):
        # If we receive more than one tensor (e.g. multiple masks), execute on each tensor and return
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            return tuple(self._ensure_safety(tensor = t) for t in tensor)
        
        if self.test_nan_inf == True:
            if isinstance(tensor, tuple):
                for t in tensor:
                    if torch.isnan(t).any() or torch.isinf(t).any():
                        raise ExclusionError("NaN or Inf found in batch.")
            else:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    raise ExclusionError("NaN or Inf found in batch.")
        return tensor

    def apply_cpu_transforms(self, tensor: torch.Tensor, target: Union[torch.Tensor, List[torch.Tensor, ], ]):
        # Resize according to desire
        if self.num_masks != 0:
            tensor = self._apply_resize(tensor)
            target = [self._apply_resize(t, interpolation = "nearest") for t in target]
        else:
            tensor = self._apply_resize(tensor)

        # Normalize according to desire
        tensor = self._apply_normalizer(tensor)

        # CPU transforms
        # If target is a mask, assume that the transforms are albumentations co-transforms
        if self.cpu_transforms is not None and self.state == "train":
            if self.num_masks != 0:
                transformed = self.cpu_transforms(tensor, masks=[target])
                tensor, target = transformed["image"], transformed["masks"]
            else:
                tensor = self.cpu_transforms(tensor)
        
        # Test for NaN/Inf, if desired
        tensor = self._ensure_safety(tensor)
        target = self._ensure_safety(target)

        # Finally, return tensor(s)
        return tensor, target

    def apply_gpu_transforms(self, tensor: torch.Tensor, target: Union[torch.Tensor, List[torch.Tensor, ], None, ] = None):

        # GPU transforms
        # If target is given, assume transforms are albumentations co-transforms and transform target
        if self.gpu_transforms is not None and self.state == "train":
            if target is not None and self.num_masks != 0:
                # Some alb-augmentations require numpy and float32 to function properly, aswell as a specific dimension order
                array_images = convert_tensor_to_opencv_array(tensor, as_type = np.float32)
                array_masks = [convert_tensor_to_opencv_array(t, as_type = np.int64) for t in target]
                transformed = self.gpu_transforms(image = array_images, masks = array_masks)
                transformed_array, transformed_target = transformed["image"], transformed["masks"]
                del(tensor)
                del(target)
                tensor = convert_opencv_array_to_tensor(transformed_array, as_type = (torch.float16 if self.cache_precision == 16 else torch.float32))
                target = [convert_opencv_array_to_tensor(t, as_type = torch.int64) for t in transformed_target]
                del(transformed_array)
                del(transformed_target)
            # If not, we expect no albumentations co-transforms, and just transform the image tensor
            else:
                tensor = self.gpu_transforms(tensor)

        # Test for NaN/Inf, if desired and necessary
        if self.gpu_transforms is not None and self.test_nan_inf is True:
            tensor = self._ensure_safety(tensor)
            if target is not None:
                target = self._ensure_safety(target)

        return tensor, target

    def __getitem__(self, idx: int, layer: int = 0):
        # If too many __getitem__ calls fail, raise an error
        if layer == 3:
            raise RuntimeError("Preventing __getitem__ loop. Something is likely wrong with the data handling.")

        # Careful load
        try:
            tensor, idx, target = self._careful_load(idx)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if self.debug is True:
                raise
            if self.verbose:
                oidx = self.subsets[self.sss][idx]
                print(f"_careful_load failed for idx {idx} (located at {self.FileList[oidx]}) with reason:", repr(e))
            tensor, idx, target = self.__getitem__(np.random.randint(0, self.__len__()), layer=layer+1)

        # Apply transforms, replace item if transforms throw error. Perform transforms on gpu is desired.
        try:
            if self.tf_device == "cpu":
                if self.num_masks != 0:
                    tensor, target = self.apply_gpu_transforms(tensor, target = target)
                else:
                    tensor, _ = self.apply_gpu_transforms(tensor)
            else:
                pass # User must manually apply transforms before or in model forward pass
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if self.debug is True:
                raise
            if self.verbose:
                oidx = self.subsets[self.sss][idx]
                print(f"_apply_transforms failed for idx {idx} (located at {self.FileList[oidx]}) with reason:", repr(e))
            tensor, idx, target = self.__getitem__(np.random.randint(0, self.__len__()), layer=layer+1)
        
        # Finally return
        return tensor, idx, target

class cacheloader(Custom_Dataset):

    """
    Pre-caches the dataset, by inheriting all relevant parameters from it and performing all required load-ops.
    """

    def __init__(self, oidxs, Files, ds):
        self.oidxs = oidxs
        self.Files = Files
        self.ds = ds

    def __call__(self):
        cached_items = []
        for i, File in enumerate(self.Files):
            oidx = self.oidxs[i]
            try:
                print(oidx)
                tensor, _, target = self.ds._load(oidx = oidx, File = File)
                tensor, target = self.ds._ensure_conformity(tensor, target)
                tensor, target = self.ds.apply_cpu_transforms(tensor, target)
                tensor = self.ds._ensure_safety(tensor)
                target = self.ds._ensure_safety(target)
                cached_items.append(((tensor.numpy(), target.numpy()), oidx))
                #cached_items.append(((tensor, target), oidx))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(repr(e))
                cached_items.append(((None, None), oidx))
        #print(f"Call complete for task {self.oidxs[0]//len(self.oidxs)}.")
        return cached_items
