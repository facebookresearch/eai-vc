# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os


def is_fbcode():
    try:
        from __manifest__ import fbmake  # noqa

        return True
    except ImportError:
        if os.path.exists(os.path.join(os.environ["HOME"], "fbsource")):
            return True
        return False


def get_buck_mode():
    from __manifest__ import fbmake

    mode = fbmake["build_mode"]
    if mode == "dev":
        return "dev-nosan"
    return mode


def get_cluster_type():
    host_name = os.uname()[1]
    if "rsc" in host_name:
        return "rsc"
    # FIXME These are bad assumptions
    elif os.path.isdir("/fsx-omnivore/"):
        return "aws"
    elif os.path.isdir("/private/home/"):
        return "fair"
    elif is_fbcode():
        return "prod"
    else:
        return "oss"
