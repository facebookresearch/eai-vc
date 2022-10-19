#!/usr/bin/env python3

import subprocess
from collections import defaultdict
from dataclasses import dataclass

_USERS = {
    "winvision": {
        "berniehuang",
        "chayryali",
        "cywu",
        "feichtenhofer",
        "haithamkhedr",
        "haoqifan",
        "huxu",
        "lyttonhao",
        "mannatsingh",
        "pdollar",
        "rbg",
        "shoubhikdn",
        "tetexiao",
        "vaibhava",
        "xinleic",
        "fduwjj",
        "mingzhe0908",
        "kumpera",
    },
    "omniscale": {"haoqifan", "kalyanv", "mannatsingh", "qduval", "vaibhava"},
}

_LIMITS = {"winvision": 1024, "omniscale": 896}

# Special case to catch jobs without a tag.
_UNCATEGORIZED_TAG = "#uncategorized"


# Handles sweep style jobs.
# E.g. 63082472_[1,5,6,7] or 63082472_[1,6-10]
def getJobArraySize(job_id):
    job_array = job_id.split("[")[1].split("]")[0]

    # Handles the case where max jobs are specified.
    # E.g. 63082472_[163-211%4]
    if "%" in job_array:
        return int(job_array.split("%")[1])

    result = 0
    for part in job_array.split(","):
        if "-" in part:
            a, b = part.split("-")
            result += len(range(int(a), int(b) + 1))
        else:
            result += 1
    return result


# Note: supports simple job arrays.
def getGpuRequest(job_id, num_nodes):
    num_gpus = int(num_nodes) * 8
    if "_[" in job_id:
        num_gpus *= getJobArraySize(job_id)
    return num_gpus


@dataclass
class JobInfoEntry:
    username: str
    jobid: str
    num_nodes: int
    jobname: str
    status: str


class SlurmUsagePerTag:
    def __init__(self, tag: str):
        self.tag = tag
        self.requested_per_user = defaultdict(int)
        self.running_per_user = defaultdict(int)
        self.total_requested = 0
        self.total_running = 0

    def parseUsage(self, jobinfo: JobInfoEntry):
        assert jobinfo.status in ["R", "CG", "PD", "PR", "S"]
        request = getGpuRequest(jobinfo.jobid, jobinfo.num_nodes)
        self.requested_per_user[jobinfo.username] += request
        self.total_requested += request
        if jobinfo.status in ["R", "CG"]:
            self.total_running += request
            self.running_per_user[jobinfo.username] += request


def querySlurm():
    # Query username, jobid, num nodes, job name, job status.
    slurm_command = 'squeue -a -o "%u,%i,%D,%j,%t" -S u,i'
    process = subprocess.run(
        slurm_command, shell=True, check=True, capture_output=True, text=True
    )
    jobs = process.stdout.split("\n")

    # Remove the first column entry and last empty entry.
    jobs.pop(0)
    jobs.pop()
    job_info = []
    for job in jobs:
        info = job.split(",", 4)
        job_info.append(JobInfoEntry(info[0], info[1], int(info[2]), info[3], info[4]))
    return job_info


def getJobTag(jobname: str):
    for tag in _USERS.keys():
        if tag.lower() in jobname.lower():
            return tag
    return _UNCATEGORIZED_TAG


def computeUsage(job_info):
    usage_per_tag = {}
    all_users = set()
    for tag, users in _USERS.items():
        usage_per_tag[tag.lower()] = SlurmUsagePerTag(tag)
        all_users.update(users)
    usage_per_tag[_UNCATEGORIZED_TAG] = SlurmUsagePerTag(_UNCATEGORIZED_TAG)

    for info in job_info:
        if info.username in all_users:
            tag = getJobTag(info.jobname)
            usage_per_tag[tag].parseUsage(info)

    return usage_per_tag


class Color:
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_usage_per_user(slurm_usage):
    print("Requested:")
    for user_id, request in sorted(slurm_usage.requested_per_user.items()):
        print(f"\t{user_id:20}\t{request}")
    print("Running:")
    for user_id, request in sorted(slurm_usage.running_per_user.items()):
        print(f"\t{user_id:20}\t{request}")


def main():
    job_info = querySlurm()
    usage_per_tag = computeUsage(job_info)
    print(
        f"\n{Color.BOLD}Please make sure to add the project tag in SLURM job name.{Color.END}"
    )
    if usage_per_tag[_UNCATEGORIZED_TAG].total_requested > 0:
        slurm_usage = usage_per_tag[_UNCATEGORIZED_TAG]
        print(
            f"\n{Color.RED}{Color.BOLD}WARNING The following GPU usage is uncategorized:{Color.END}"
        )
        print_usage_per_user(slurm_usage)
    for tag in _USERS.keys():
        slurm_usage = usage_per_tag[tag]
        running_requested = _LIMITS[tag] - slurm_usage.total_running
        remaining_requested = _LIMITS[tag] - slurm_usage.total_requested
        print(f"\n{Color.BOLD}Usage for tag {tag}:{Color.END}")
        print(f"Total GPUs requested: {slurm_usage.total_requested}")
        print(f"Total GPUs running: {slurm_usage.total_running}")
        print(
            f"Available GPUs counting requested (please also check #uncategorized section): {remaining_requested}"
        )
        print(
            f"Available GPUs counting only running (please also check #uncategorized section): {running_requested}"
        )
        print("GPU request per user:")
        print_usage_per_user(slurm_usage)


if __name__ == "__main__":
    main()
