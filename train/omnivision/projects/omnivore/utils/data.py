import getpass
import hashlib
import json
import logging
import os.path
import pickle
import shutil
from functools import partial
from multiprocessing import shared_memory
from typing import Any, List, Tuple, Union, Optional

import numpy as np
from iopath.common.file_io import g_pathmgr
from omnivision.trainer.distributed import is_local_primary, is_torch_dataloader_worker
from omnivision.utils.distributed import barrier, broadcast_object
from PIL import Image


class IdentityTransform:
    def __call__(self, x: Any) -> Any:
        return x


def to_list(elem):
    return list(elem)


# copied from vissl.data.data_helper
def get_mean_image(crop_size: Union[Tuple, int]):
    """
    Helper function that returns a gray PIL image of the size specified by user.
    Args:
        crop_size (tuple, or int): used to generate (crop_size[0] x crop_size[1] x 3) image
            in the case of a tuple of (crop_size, crop_size, 3) image in case of int.
    Returns:
        img: PIL Image
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    img = Image.fromarray(
        128 * np.ones((crop_size[0], crop_size[1], 3), dtype=np.uint8)
    )
    return img


def list_of_paths_to_path(path_list: List[str]):
    path_exists = False
    for idx, path in enumerate(path_list):
        if g_pathmgr.exists(path):
            path_exists = True
            break
    if path_exists is False:
        path = None
    return path_exists, path, idx


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def pickle_load(path):
    with g_pathmgr.open(path, "rb") as fh:
        data = pickle.load(fh)
    return data


def numpy_load(path):
    with g_pathmgr.open(path, "rb") as fh:
        data = np.load(fh)
    return data


def json_load(path):
    with g_pathmgr.open(path, "r") as fh:
        data = json.load(fh)
    return data


def smart_copy_path(path: str, dst_dir: str):
    """
    Performs the following
        Copy path if doesn't exist
        We compute a hash of the full path to avoid cases where two files have the same basenames
    Returns
        Path copied over into dst_dir
    """
    base_file = os.path.basename(path)
    file_ext = os.path.splitext(base_file)[1]
    path_hash = hashlib.md5(path.encode("utf8")).hexdigest()
    dst_file = os.path.join(dst_dir, path_hash + file_ext)
    try:
        if not g_pathmgr.isfile(dst_file):
            shutil.copyfile(path, dst_file)
        return dst_file
    except (OSError, shutil.SameFileError):
        # failed to copy
        logging.warning(f"Failed to copy {path} to {dst_file}")
        return path


def get_local_dst_dir_to_copy(local_dir_basename="data"):
    """
    Gets a local dir to copy to
    """
    if "SLURM_JOBID" in os.environ:
        job_id = os.environ["SLURM_JOBID"]
        local_dir = f"/scratch/slurm_tmpdir/{job_id}/{local_dir_basename}"
    else:
        local_dir = f"/tmp/{os.environ['USER']}/{local_dir_basename}"

    if not g_pathmgr.isdir(local_dir):
        g_pathmgr.mkdirs(local_dir)

    return local_dir


FILE_EXT_TO_HANDLER = {".pkl": pickle_load, ".npy": numpy_load, ".json": json_load}


class FileLoader:
    @staticmethod
    def load(path_list: List[str], file_handler=None, return_idx=True, sub_key=None):
        path_exists, path, idx = list_of_paths_to_path(path_list)
        if not path_exists:
            raise ValueError(f"No path exists in {path_list}")
        if file_handler is None:
            _, ext = os.path.splitext(path)
            file_handler = FILE_EXT_TO_HANDLER[ext]
        arr = file_handler(path)
        if sub_key is not None:
            arr = arr[sub_key]
        if return_idx:
            return arr, idx
        return arr


class SharedMemoryNumpyLoader:
    """
    WARN: A referenced to this object needs to be preserved till
    the returned np array is being used. This uses collective
    operations.
    """

    def __init__(self):
        self.sm = None
        self.sm_name = None

    def load(
        self, path_list: List[str], every_k_images: Optional[int] = None
    ) -> np.ndarray:
        """Attempts to load data from a list of paths. Each element is tried (in order)
        until a file that exists is found. That file is then used to read the data.
        """
        if self.sm is not None:
            raise RuntimeError("Cannot load multiple objects with the same loader")

        path_exists, path, idx = list_of_paths_to_path(path_list)

        if not path_exists:
            raise ValueError(f"No path exists in {path_list}")

        self.sm_name = (
            "".join([x if x.isalnum() else "_" for x in path]) + f"_{getpass.getuser()}"
        )

        # we only read from local rank 0 parent process on a machine
        # all other GPU parent processes and dataloaders read from shared memory
        if is_local_primary() and not is_torch_dataloader_worker():
            # this is the local rank 0 process
            arr = load_file(path, every_k_images=every_k_images)
            assert isinstance(
                arr, np.ndarray
            ), f"arr is not an ndarray. found {type(arr)}"
            logging.info(f"Moving data files to shared memory: {self.sm_name}")
            try:
                sm = shared_memory.SharedMemory(
                    name=self.sm_name, create=True, size=arr.nbytes
                )
            except FileExistsError:
                logging.info(
                    "Shared memory already exists, closing it out and recreating"
                )
                sm_old = shared_memory.SharedMemory(name=self.sm_name, create=False)
                sm_old.close()
                sm_old.unlink()
                sm = shared_memory.SharedMemory(
                    name=self.sm_name, create=True, size=arr.nbytes
                )
            sm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=sm.buf)
            sm_arr[:] = arr[:]
            # barrier for all (non-dataloader) proceses to ensure the data is
            # available on all GPUs
            barrier()
            broadcast_object(sm_arr.shape)  # arr_shape
            broadcast_object(sm_arr.dtype)  # arr_type
        else:
            if not is_torch_dataloader_worker():
                # parent process on a GPU which isn't local rank 0; wait for barrier
                barrier()
                arr_shape = broadcast_object(None)
                arr_dtype = broadcast_object(None)
            logging.info(f"Loading data files from shared memory: {self.sm_name}")
            sm = shared_memory.SharedMemory(name=self.sm_name, create=False)
            sm_arr = np.ndarray(shape=arr_shape, dtype=arr_dtype, buffer=sm.buf)
        # need to keep a reference to the shared memory otherwise it will get
        # garbage collected and result in a segfault
        self.sm = sm
        return sm_arr, idx

    def __del__(self):
        # FIXME: this doesn't seem to be working on the FAIR cluster
        if self.sm is None:
            return
        self.sm.close()
        if is_local_primary() and not is_torch_dataloader_worker():
            logging.info(f"Unlinking shared memory: {self.sm_name}")
            self.sm.unlink()


# Copied from vissl.utils.io
def load_file(filename, mmap_mode=None, every_k_images=None):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    logging.info(f"Loading data from file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding="latin1")
    elif file_ext == ".npy":
        # keeping allow_pickle=False which might be needed for object arrays
        # (eg for multilabel datasets [[3,4], [1], [1,2,3], ...])
        # Loading such non native numpy arrays can be problematic though
        # as it would copy the array in all the workers (shared mem doesn't work)
        # so avoid if possible.
        np_load_fn = partial(np.load, encoding="latin1")
        if mmap_mode:
            np_load_fn = partial(np_load_fn, mmap_mode=mmap_mode)
            try:
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np_load_fn(fopen)
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without PathManager"
                )
                data = np_load_fn(filename)
                logging.info("Successfully loaded without PathManager")
            except Exception:
                logging.info("Could not mmap without PathManager. Trying without mmap")
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np_load_fn(fopen)
        else:
            with g_pathmgr.open(filename, "rb") as fopen:
                data = np_load_fn(fopen)
    elif file_ext == ".json":
        with g_pathmgr.open(filename, "r") as fopen:
            data = json.loads(fopen.read())
    elif file_ext == ".txt":
        with g_pathmgr.open(filename, "r") as fopen:
            data = np.array(
                fopen.read().splitlines()
            )  # use np.ndarray for shared memory
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    if every_k_images is not None:
        data = data[::every_k_images]
    return data


def load_file_from_list(file_list, mmap_mode=None):
    for path in file_list:
        if g_pathmgr.exists(path):
            return load_file(path, mmap_mode)
            break
    raise Exception(f"None of the paths exist in {file_list}")


# VIDEO UTILS


def paths_to_clips(paths, every_k=None, min_num_frames=None):
    logging.info("Converting paths to clips...")

    every_k = 1 if every_k is None else every_k
    min_num_frames = 2 if min_num_frames is None else min_num_frames

    # save start and end index for each video in path list
    videos = {}
    for idx, path in enumerate(paths):
        vidname = os.path.split(path)[0]
        if vidname not in videos:
            videos[vidname] = []
        else:
            # check that indices are consecutive
            assert videos[vidname][-1] == idx - 1
        videos[vidname].append(idx)

    clips = []
    for k, v in videos.items():
        # check number of frames in clip
        frames = [i for i in range(0, len(v), every_k)]
        if len(frames) < min_num_frames:
            logging.warning(
                f"Dropping vid {k} with length {len(frames)} < {min_num_frames}"
            )
            continue
        clips.append((v[0], v[-1]))

    clips = np.array(clips)

    logging.info("Converting paths to clips... done!")
    return clips


def _create_shared_memory(sm_name, arr):
    assert isinstance(arr, np.ndarray), f"arr is not an ndarray. found {type(arr)}"
    logging.info(f"Moving data files to shared memory: {sm_name}")
    try:
        sm = shared_memory.SharedMemory(name=sm_name, create=True, size=arr.nbytes)
    except FileExistsError:
        logging.info("Shared memory already exists, closing it out and recreating")
        sm_old = shared_memory.SharedMemory(name=sm_name, create=False)
        sm_old.close()
        sm_old.unlink()
        sm = shared_memory.SharedMemory(name=sm_name, create=True, size=arr.nbytes)
    sm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=sm.buf)
    sm_arr[:] = arr[:]
    # barrier for all (non-dataloader) process to ensure the data is
    # available on all GPUs
    barrier()
    broadcast_object(sm_arr.shape)  # arr_shape
    broadcast_object(sm_arr.dtype)  # arr_type
    return sm, sm_arr


def _wait_for_shared_memory(sm_name):
    if not is_torch_dataloader_worker():
        # parent process on a GPU which isn't local rank 0; wait for barrier
        barrier()
        arr_shape = broadcast_object(None)
        arr_dtype = broadcast_object(None)
    logging.info(f"Loading data files from shared memory: {sm_name}")
    sm = shared_memory.SharedMemory(name=sm_name, create=False)
    sm_arr = np.ndarray(shape=arr_shape, dtype=arr_dtype, buffer=sm.buf)
    return sm, sm_arr


class SharedMemoryVideoLoader:
    """
    WARN: A referenced to this object needs to be preserved till
    the returned np array is being used. This uses collective
    operations.
    """

    def __init__(self):
        self.paths_sm, self.paths_sm_name = None, None
        self.clips_sm, self.clips_sm_name = None, None

    def load(
        self,
        path_list: List[str],
        every_k: int,
        min_num_frames: int,
        repeat_factor: int,
    ) -> np.ndarray:
        """Attempts to load data from a list of paths. Each element is tried (in order)
        until a file that exists is found. That file is then used to read the data.
        """
        if self.paths_sm is not None or self.clips_sm is not None:
            raise RuntimeError("Cannot load multiple objects with the same loader")

        path_exists, path, _ = list_of_paths_to_path(path_list)

        if not path_exists:
            raise ValueError(f"No path exists in {path_list}")

        base_sm_name = (
            "".join([x if x.isalnum() else "_" for x in path]) + f"_{getpass.getuser()}"
        )
        self.paths_sm_name = base_sm_name + "_paths"
        self.clips_sm_name = base_sm_name + "_clips"

        # we only read from local rank 0 parent process on a machine
        # all other GPU parent processes and dataloaders read from shared memory
        if is_local_primary() and not is_torch_dataloader_worker():
            # this is the local rank 0 process
            paths_arr = load_file(path)
            clips_arr = paths_to_clips(paths_arr, every_k, min_num_frames)
            clips_arr = np.tile(clips_arr, (repeat_factor, 1))

            paths_sm, paths_sm_arr = _create_shared_memory(
                self.paths_sm_name, paths_arr
            )
            clips_sm, clips_sm_arr = _create_shared_memory(
                self.clips_sm_name, clips_arr
            )
        else:
            paths_sm, paths_sm_arr = _wait_for_shared_memory(self.paths_sm_name)
            clips_sm, clips_sm_arr = _wait_for_shared_memory(self.clips_sm_name)
        # need to keep a reference to the shared memory otherwise it will get
        # garbage collected and result in a segfault
        self.paths_sm = paths_sm
        self.clips_sm = clips_sm
        return paths_sm_arr, clips_sm_arr

    def __del__(self):
        # FIXME: this doesn't seem to be working on the FAIR cluster
        if self.paths_sm is None and self.clips_sm is None:
            return
        if self.paths_sm is not None:
            self.paths_sm.close()
            if is_local_primary() and not is_torch_dataloader_worker():
                logging.info(f"Unlinking shared memory: {self.paths_sm_name}")
                self.paths_sm.unlink()
        if self.clips_sm is not None:
            self.clips_sm.close()
            if is_local_primary() and not is_torch_dataloader_worker():
                logging.info(f"Unlinking shared memory: {self.clips_sm_name}")
                self.clips_sm.unlink()
