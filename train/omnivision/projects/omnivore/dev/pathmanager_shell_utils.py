"""
Functions to run iopath pathmanager utilities through shell commands
(hence does not need to be compiled with buck)
"""
import os
import subprocess
import tempfile
from typing import List

try:
    from iopath.common.file_io import g_pathmgr
    from iopath.fb.manifold import ManifoldPathHandler  # noqa

    PTHMGR_FB_FOUND = True
except ImportError:
    PTHMGR_FB_FOUND = False

MANIFOLD_PREFIX = "manifold://"


def split_manifold_path(fpath):
    dirname = fpath[len(MANIFOLD_PREFIX) :]
    bucket, path = dirname.split("/", 1)
    return bucket, path


def manifold_call(cmd, dirname):
    assert dirname.startswith(MANIFOLD_PREFIX)
    dirname = dirname[len(MANIFOLD_PREFIX) :]
    comp_proc = subprocess.run(
        f"manifold {cmd} {dirname}", shell=True, capture_output=True
    )
    return (dirname, comp_proc.returncode, comp_proc.stdout.decode().strip())


def PathManager_exists(full_dirname: str):
    if PTHMGR_FB_FOUND:
        return g_pathmgr.exists(full_dirname)

    if full_dirname.startswith(MANIFOLD_PREFIX):
        dirname, retval, output = manifold_call("exists", full_dirname)
        if output == f"{dirname} EXISTS":
            assert retval == 0
            return True
        assert retval == 1
        return False
    elif os.path.exists(full_dirname):
        return True
    else:
        return False


def PathManager_mkdirs(full_dirname: str):
    if PTHMGR_FB_FOUND:
        return g_pathmgr.mkdirs(full_dirname)

    if full_dirname.startswith(MANIFOLD_PREFIX):
        _, retval, output = manifold_call("mkdirs", full_dirname)
        if len(output) == 0:
            return retval
        raise ValueError(f"Exception: {output}")
    else:
        # Normal folder
        subprocess.run(f"mkdir -p {full_dirname}", shell=True)


def PathManager_rm(full_dirname: str):
    if PTHMGR_FB_FOUND and g_pathmgr.isfile(full_dirname):
        return g_pathmgr.rm(full_dirname)

    if full_dirname.startswith(MANIFOLD_PREFIX):
        _, retval, output = manifold_call("rm", full_dirname)
        if len(output) == 0:
            return retval
        raise ValueError(f"Exception: {output}")
    else:
        # Normal folder
        subprocess.run(f"rm -r {full_dirname}", shell=True)


def PathManager_copy(
    src: str, dst: str, overwrite: bool = False, threads: int = 0
) -> bool:
    """
    Returns:
        True on success
    """
    if PTHMGR_FB_FOUND:
        return g_pathmgr.copy(src, dst, overwrite=overwrite)

    if src.startswith(MANIFOLD_PREFIX) and dst.startswith(MANIFOLD_PREFIX):
        # It is a copy operation
        overwrite_flag = "--overwrite" if overwrite else ""
        assert threads == 0, "Not implemented yet"
        src_bucket, src_path = split_manifold_path(src)
        dst_bucket, dst_path = split_manifold_path(dst)
        assert src_bucket == dst_bucket
        return (
            subprocess.run(
                f"manifold copy {src_bucket} {src_path} {dst_path} {overwrite_flag}",
                shell=True,
            ).returncode
            == 0
        )
    elif dst.startswith(MANIFOLD_PREFIX):
        # put operation
        threads_flag = f"--threads {threads}"
        overwrite_flag = "--overwrite" if overwrite else ""
        return (
            subprocess.run(
                f"manifold put {src} {dst[len(MANIFOLD_PREFIX) :]} "
                f"{overwrite_flag} {threads_flag} ",
                shell=True,
            ).returncode
            == 0
        )
    elif src.startswith(MANIFOLD_PREFIX):
        # Get operation
        threads_flag = f"--jobs {threads} --parallel"
        overwrite_flag = "--overwrite_local_path" if overwrite else ""
        return (
            subprocess.run(
                f"manifold get {src[len(MANIFOLD_PREFIX) :]} {dst} "
                f"{overwrite_flag} {threads_flag} ",
                shell=True,
            ).returncode
            == 0
        )
    else:
        # Normal folder
        assert not overwrite, "Not implemented yet"
        assert threads == 0, "Not implemented yet"
        return subprocess.run(f"cp -r {src} {dst}", shell=True).returncode == 0


def PathManager_readlines(fpath: str):
    if PTHMGR_FB_FOUND:
        with g_pathmgr.open(fpath, "r") as fin:
            return fin.readlines()

    is_tmp_fpath = False
    if fpath.startswith(MANIFOLD_PREFIX):
        fpath_to_read = tempfile.NamedTemporaryFile(delete=False).name
        PathManager_copy(fpath, fpath_to_read, overwrite=True, threads=10)
    else:
        # Normal file
        fpath_to_read = fpath
    with open(fpath_to_read, "r") as fin:
        output = fin.readlines()
    if is_tmp_fpath:
        os.remove(fpath_to_read)
    return output


def PathManager_ls(dpath: str) -> List[str]:
    if PTHMGR_FB_FOUND:
        return g_pathmgr.ls(dpath)
    if dpath.startswith(MANIFOLD_PREFIX):
        dirname, retval, output = manifold_call("ls", dpath)
        # Remove the filesize information
        return [el.strip().split(" ", 1)[1] for el in output.splitlines()]
    else:
        # Likely never needed... else implement when it is
        raise NotImplementedError()
