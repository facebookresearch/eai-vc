## Using Cloud TPUs with PyTorch XLA (MAE pretraining as an example)

***Note: Please refer to the [FAIR Cloud TPU Onboarding Guide](https://github.com/fairinternal/fair_gcp_tpu_docs/blob/main/README.md) for the most up-to-date tips on PyTorch XLA and TPU usage.***

This document provides a guide for Cloud TPU experiments with PyTorch XLA and TPU VMs, using MAE pretraining as an example.

Table of contents
* [Step 1: Creating your TPU VM](#step-1-creating-your-tpu-vm)
  * [Step 1.1: Installing Google Cloud SDK and account](#step-11-installing-google-cloud-sdk-and-account)
  * [Step 1.2: Making a TPU VM startup script](#step-12-making-a-tpu-vm-startup-script)
  * [Step 1.3: Allocating the TPU VM](#step-13-allocating-the-tpu-vm)
  * [Step 1.4: Logging into the TPU VM](#step-14-logging-into-the-tpu-vm)
* [Step 2: Setting up codebase and datasets](#step-2-setting-up-codebase-and-datasets)
  * [Step 2.1: Setting up codebase](#step-21-setting-up-codebase)
  * [Step 2.2: Setting up datasets](#step-22-setting-up-datasets)
* [Step 3: Running experiments with PyTorch XLA](#step-3-running-experiments-with-pytorch-xla)
  * [Running MAE pretraining](#running-mae-pretraining)
  * [Speed](#speed)
  * [Troubleshooting](#troubleshooting)
  * [Debugging on a single TPU VM node (instead of a pod)](#debugging-on-a-single-tpu-vm-node-instead-of-a-pod)
  * [Other PyTorch XLA examples](#other-pytorch-xla-examples)
  * [Developing new models](#developing-new-models)
* [Storage options](#storage-options)
  * [Filestore NFS](#filestore-nfs)
  * [Google Cloud Storage](#google-cloud-storage)
  * [Persistent disk](#persistent-disk)
* [Useful commands for TPU VMs](#useful-commands-for-tpu-vms)
  * [Installing a dependency or running a command on all VM nodes](#installing-a-dependency-or-running-a-command-on-all-vm-nodes)
  * [Copying files from or to a TPU VM](#copying-files-from-or-to-a-tpu-vm)
  * [Setting up VSCode connection](#setting-up-vscode-connection)
  * [Deleting a TPU VM](#deleting-a-tpu-vm)
  * [Using a remote coordinator VM to guard against maintenance events](#using-a-remote-coordinator-vm-to-guard-against-maintenance-events)
  * [Checking TPU status (busy/idle)](#checking-tpu-status-busyidle)
  * [Speed profiling and optimization](#speed-profiling-and-optimization)

---

### Step 1: Creating your TPU VM

[TPU VMs](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) are now the recommended way to use cloud TPUs. They are directly attached to TPU chips and are faster than using standalone compute VM with TPU nodes. In the example below, we will first create a startup script and then allocate the TPU VM with it.

#### Step 1.1: Installing Google Cloud SDK and account

To use Google Cloud TPUs, first install the Google Cloud SDK and log into your Google Cloud account, and project following the instructions in https://cloud.google.com/sdk/docs/quickstart.

(If you are a member of FAIR, you can follow the [FAIR GCP login guide](https://fburl.com/wiki/7kswgk2a); replace the Google Cloud SDK url in this guide with the latest one [here](https://cloud.google.com/sdk/docs/quickstart#installing_the_latest_version).)

#### Step 1.2: Making a TPU VM startup script

First, let's create a TPU VM *startup script* by saving the content below to a file `tpu_start_mae.sh`. The startup script contains the command to run when setting up a new TPU VM node and should contain all the dependencies we want to install.

Note: this script mounts Ronghang's Filestore NFS directory (`10.89.225.82:/mmf_megavlt`, where ImageNet is stored) to `/checkpoint`. **You should create your own Filestore NFS directory [here](https://console.cloud.google.com/filestore/instances?authuser=1) and modify the startup script accordingly.** (One should create the NFS directory in `europe-west4-a` location where we will create our TPUs below).

```
# (save this content to a file "tpu_start_mae.sh")

# install all the dependencies needed for training
sudo pip3 install timm==0.4.12  # use timm 0.4.12 in MAE pretraining for compatibility with PyTorch 1.10

# !!! this script mounts Ronghang's NFS directory (`10.89.225.82:/mmf_megavlt`) to `/checkpoint`.
# !!! You should create your own NFS directory in https://console.cloud.google.com/filestore/instances?authuser=1
# !!! and modify the startup script accordingly
sudo apt-get -y update
sudo apt-get -y install nfs-common
SHARED_FS=10.89.225.82:/mmf_megavlt
MOUNT_POINT=/checkpoint
sudo mkdir -p $MOUNT_POINT
for i in $(seq 10); do  # try mounting 10 times to avoid transient NFS mounting failures
  ALREADY_MOUNTED=$(($(df -h | grep $SHARED_FS | wc -l) >= 1))
  (($ALREADY_MOUNTED == 1)) || sudo mount $SHARED_FS $MOUNT_POINT
done
sudo chmod go+rw $MOUNT_POINT
```

#### Step 1.3: Allocating the TPU VM

Now, you can create your TPU VM with the above startup script (see [Cloud TPU PyTorch quickstart](https://cloud.google.com/tpu/docs/pytorch-quickstart-tpu-vm) for more details).

In the example below, we will create a [v3-128 TPU pod](https://cloud.google.com/tpu/docs/types-zones#europe) as used in the MAE paper. Here we create our TPUs in `europe-west4-a` location based on our TPU [quota](https://console.cloud.google.com/iam-admin/quotas?authuser=1&project=fair-infra3f4ebfe6).

```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to another name you like
ZONE=europe-west4-a  # a location where we have available TPU quota
ACCELERATOR_TYPE=v3-128
STARTUP_SCRIPT=/Users/ronghanghu/workspace/gcp_scripts/tpu_start_mae.sh  # !!! change to your startup script path

RUNTIME_VERSION=tpu-vm-pt-1.10  # this is the runtime we use for PyTorch XLA (it contains PyTorch 1.10)

# create a TPU VM (adding `--reserved` to create reserved TPUs)
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --zone ${ZONE} \
  --accelerator-type ${ACCELERATOR_TYPE} --reserved \
  --version ${RUNTIME_VERSION} \
  --metadata-from-file=startup-script=${STARTUP_SCRIPT}
```

#### Step 1.4: Logging into the TPU VM

Now we can log into the TPU VM we just created.
```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
# it takes a while for the SSH to work after creating TPU VM
# if this command fails, just retry it
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker 0
```
Here `--worker 0` means that we are going to log into the first VM node (a v3-128 TPU pod has 16 VM nodes, and each node is connected to 8 TPU cores). Note: it may take a few minutes for the setup script to finish (and it could still be running in the background after you log in). *So if you don't see your NFS directory (mounted to `/checkpoint` above), it should appear in one or two minutes.* (In case it doesn't appear after a while, see "Troubleshooting".)

After logging into TPU VM, now we can set up the codebase and datasets for our experiments.

---

### Step 2: Setting up codebase and datasets

#### Step 2.1: Setting up codebase

On the TPU VM after logging in, we should store all the codebase in **a shared NFS directory** so that the same repo can be accessed from all VM nodes.

In the example below, we clone the TPU-compatible MAE repo under `/checkpoint/ronghanghu/workspace` as follows. Note: [install your GitHub ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) on your TPM VM before cloning the repo.
```
# (run on your TPU VM)

# !!! this should be under a shared NFS directory (you can change to a different path)
# !!! install your GitHub ssh key on your TPM VM before cloning it
WORKSPACE_DIR=/checkpoint/ronghanghu/workspace
mkdir -p $WORKSPACE_DIR && cd $WORKSPACE_DIR
git clone git@github.com:fairinternal/mae_tpu.git -b mae_pretraining_pytorch_xla
```

#### Step 2.2: Setting up datasets

On the TPU VM, download datasets to **a shared NFS directory** so that the data can be accessed from all VM nodes.

In the example below, the [ImageNet-1k](https://image-net.org/) data is downloaded to `/checkpoint/imagenet-1k/`, which should have the following structure (the validation images to labeled subfolders, following the [PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet#requirements)).
```
/checkpoint/imagenet-1k
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```

Now your TPU VM is set up and we can run experiments on it.

---

### Step 3: Running experiments with PyTorch XLA

#### Running MAE pretraining

Before running any experiments, first set up the gcloud ssh configuration on your TPM VM as follows (*only need to do it once*):
```
# (run on your TPU VM)

cd ${HOME} && gcloud compute config-ssh --quiet
```

Now we can run our experiments. To pretrain ViT-Large with MAE, run the following **in a tmux sesssion** (note that the first few iterations are very slow due to compilation):
```
# (run on your TPU VM, preferably in a tmux session)

MAE_PATH=/checkpoint/ronghanghu/workspace/mae_tpu  # where the repo is cloned above
IMAGENET_DIR=/checkpoint/imagenet-1k/  # where ImageNet-1k is downloaded above
SAVE_DIR=/checkpoint/ronghanghu/mae_save/vitl_800ep  # a place to save checkpoints (should be under NFS)
MODEL=mae_vit_large_patch16
EPOCH=800
TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
BATCH_SIZE_PER_TPU=32  # 4096 (total batch size) // 128 (tpu cores)

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # a workaround for NFS UIDs (see "Troubleshooting")
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist \
  --tpu=${TPU_NAME} --restart-tpuvm-pod-server \
  --env XRT_MESH_CONNECT_WAIT=1200 --env PYTHONUNBUFFERED=1 -- \
python3 ${MAE_PATH}/main_pretrain.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE_PER_TPU} \
    --model ${MODEL} \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs ${EPOCH} \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --num_workers 8 \
    --use_xla --resume automatic \
    2>&1 | tee $SAVE_DIR/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
```
Here `--use_xla` runs the script in XLA mode, and `--resume automatic` automatically searches and loads the last checkpoint. See [`PRETRAIN.md`](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md) for the details of all other parameters. The stdout and stderr outputs are saved under `$SAVE_DIR/stdout_stderr_*.log`. Note that the training processes need to be launched on all VM nodes in a TPU pod (e.g. a v3-128 TPU pod has 16 nodes with 8 TPU cores attached to each node) and `torch_xla.distributed.xla_dist` is used to spawn the training process on all the VM nodes.

This PyTorch XLA TPU implementation gets 85.557% [fine-tuning](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md) accuracy by pre-training ViT-Large for 800 epochs (85.4% in paper Table 1d with TF/TPU). Note that we copy the TPU-pretrained checkpoints and run fine-tuning evaluation on GPUs (see the "Copying files from or to a TPU VM" section below for how to copy files with gcloud scp).

#### Speed

On a v3-128 TPU pod, the MAE pretraining speed of this codebase for ViT-L is roughly 0.46 sec/iter. This corresponds to 31.9 hours for 800 epochs. (Note that the first few iterations are very slow due to XLA compilation). You can also try out a v3-256 TPU pod to get higher speed (use `ACCELERATOR_TYPE=v3-256` when creating the TPU VM and set `BATCH_SIZE_PER_TPU=16` in the commands to keep the 4096 effective batch size). Also, see "Speed profiling and optimization" below for more performance tips.

#### Troubleshooting

1. Note that in a few rare cases, the TPU VM startup script can fail to set up the NFS directory on all TPU VM nodes. So if your training hangs around "effective batch size: 4096" and you have the following error in your log (as saved to `$SAVE_DIR/stdout_stderr_*.log` above)
```
2022-01-09 03:04:10 10.164.0.64 [14] python3: can't open file '/checkpoint/ronghanghu/workspace/mae_tpu/main_pretrain.py': [Errno 2] No such file or directory
```
it shows that the code cannot be accessed from node 14 (shown as `[14]`). To fix it, log into the problematic node via `gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker 14` and manually mount NFS disk as in the startup script. A simple way to re-run the NFS mounting on all worker nodes is to connect to all nodes in gcloud SSH via `--worker all` as follows
```
TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
SHARED_FS=10.89.225.82:/mmf_megavlt  # !!! change to your NFS directory
MOUNT_POINT=/checkpoint  # !!! change to your mounting point

gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --worker all \
  --command "(((\$(df -h | grep $SHARED_FS | wc -l) >= 1)) && echo NFS already mounted on \$(hostname)) || (sudo mkdir -p $MOUNT_POINT && sudo mount $SHARED_FS $MOUNT_POINT && sudo chmod go+rw $MOUNT_POINT && echo mounted NFS on \$(hostname))"
```

2. This TPU pretraining code is tested around 01/20/2022. If you are running it at a later date and the pretraining hangs, it may be helpful to install the latest libtpu to match the TPU environment as follows.
```
# (run on your TPU VM)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist --tpu=${TPU_NAME} -- \
  sudo pip3 install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.devYYYYMMDD-py3-none-any.whl
```
where `YYYYMMDD` is a date string such as `20220119`. (Note that I saw a speed regression in the nightly version of `20220119` on v3-128 that increases the per-iter time from 0.46 sec to 0.6 sec, but this regression doesn't happen on v3-256.)

3. Sometimes if a training crashes, there can be some remaining Python processes on the VM nodes that prevent new training from being launched. One can kill them as follows.
```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --command "pkill python" --worker all
```

4. The NFS storage may suffer from inconsistent UIDs when using TPU VMs in a pod that can cause permission issues during training. Each VM node has a different user-name to UID mapping in its "/etc/passwd" file. For example, on one node in a TPU pod I have `ronghanghu:x:2017:2017::/home/ronghanghu:/bin/bash` while on another node I have `ronghanghu:x:2002:2002::/home/ronghanghu:/bin/bash`. This UID inconsistency on TPU VMs causes issues when trying to access the same NFS directory from multiple nodes: The same directory would show different owners on different nodes, so standard Linux permissions don't apply consistently across nodes. For example, our training process in one node cannot write to the log directory created by another node, which can cause permission denied errors or crashes during training.    
My workaround is to set everything in our saving directory to 777 permission before starting the training, as in `sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR` above, which works quite well for me. If your training script involves writing to sub-directories (e.g. for logging), you can first create these sub-directories before the training and set their permission to 777. (Another option to fix the UID inconsistency is to use [OS login](https://cloud.google.com/compute/docs/oslogin).)

5. In the `torch_xla.distributed.xla_dist` command above, the environment variables in the host TPU VM (where this command is launched) are not automatically carried over in the training process. One can specify environment variables via one or multiple `--env` arguments. For example, use `--env VAR1=XXX --env VAR2=YYY --env VAR3=ZZZ` to pass three environment variables. It is usually helpful to also add `--restart-tpuvm-pod-server` to restart the XRT (XLA Runtime) server for each training run.

6. Some recent versions of `gcloud` cannot be executed from an NFS directory. Hence if you are seeing errors like `cannot open path of the current working directory: Permission denied`, it's likely that you are running `torch_xla.distributed.xla_dist` from an NFS directory. You can switch back via `cd ${HOME}` and launch experiments from there. In some cases, it is helpful to add your development code path to the `PYTHONPATH` environment variable via `--env PYTHONPATH=/your/code/path` in `torch_xla.distributed.xla_dist` so that your `python3` command can be started from any directory.

7. See more debugging and troubleshooting tips here: https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md.

#### Debugging on a single TPU VM node (instead of a pod)

As an alternative to pod training (launching on all the 128 TPU cores in a v3-128 pod), one can also launch on a single TPU VM node with 8 TPU cores (i.e. v3-8) for development and debugging. For this purpose, you can set `XRT_TPU_CONFIG="localservice;0;localhost:51011"` and run the python script directly (without using `torch_xla.distributed.xla_dist`). One can also set `PT_XLA_DEBUG=1` to ask PyTorch XLA to print more information for debugging.

For example, to train MAE on a single TPU VM node with 8 TPU cores for debugging purposes:
```
# (run on your TPU VM, preferably in a tmux session)

export XRT_TPU_CONFIG="localservice;0;localhost:51011"  # using a local XRT server for 8 TPU cores
export PT_XLA_DEBUG=1  # ask PyTorch XLA to print more information for debugging

MAE_PATH=/checkpoint/ronghanghu/workspace/mae_tpu  # where the repo is cloned above
IMAGENET_DIR=/checkpoint/imagenet-1k/  # where ImageNet-1k is downloaded above
SAVE_DIR=/checkpoint/ronghanghu/mae_save/vitl_800ep_debug  # a place to save checkpoints (should be under NFS)
MODEL=mae_vit_large_patch16
EPOCH=800
ACCUM_ITER=16  # gradient accumulation to get a larger effective batch size
BATCH_SIZE_PER_TPU=32  # 4096 (total batch size) // 8 (tpu cores) // 16 (accumulation steps)

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # a workaround for NFS UIDs (see "Troubleshooting")
python3 -u ${MAE_PATH}/main_pretrain.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE_PER_TPU} --accum_iter ${ACCUM_ITER} \
    --model ${MODEL} \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs ${EPOCH} \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --num_workers 8 \
    --use_xla --resume automatic \
    2>&1 | tee $SAVE_DIR/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
```

#### Other PyTorch XLA examples

Other PyTorch XLA codebases can be launched in a similar way. For example, one can run the PyTorch XLA [ResNet-50 ImageNet classification example](https://cloud.google.com/tpu/docs/tutorials/resnet-pytorch) by first setting up the code
```
WORKSPACE_DIR=/checkpoint/ronghanghu/workspace  # this should be under a shared NFS directory
mkdir -p $WORKSPACE_DIR && cd $WORKSPACE_DIR
git clone git@github.com:pytorch/xla.git
```
and then launching the training on the TPU pod via `torch_xla.distributed.xla_dist`:
```
TPU_NAME=mae-tpu-128
CODE_PATH=/checkpoint/ronghanghu/workspace/xla/test/test_train_mp_imagenet.py
IMAGENET_DIR=/checkpoint/imagenet-1k/  # where ImageNet-1k is downloaded above
SAVE_DIR=/checkpoint/ronghanghu/resnet50_in1k  # a place to save checkpoints (should be under NFS)

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # a workaround for NFS UIDs (see "Troubleshooting")
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist \
  --tpu=${TPU_NAME} --restart-tpuvm-pod-server \
  --env XRT_MESH_CONNECT_WAIT=1200 --env PYTHONUNBUFFERED=1 -- \
python3 ${CODE_PATH} \
  --datadir $IMAGENET_DIR \
  --drop_last \
  --model=resnet50 \
  --batch_size=32 \
  --lr=1.6 \
  --momentum=0.9 \
  --num_epochs=100 \
  --lr_scheduler_divisor=10 \
  --lr_scheduler_divide_every_n_epochs=30 \
  --log_steps=100 \
  2>&1 | tee $SAVE_DIR/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
```
Note that `--batch_size=32` above is the batch size per TPU core, not the total batch size (the total batch size is `32 * num-of-tpu-cores`; a v3-128 TPU pod has 128 cores). The hyperparameters above such as learning rate are just for illustration and are not optimal for ResNet-50.

#### Developing new models

As a useful starting guide to develop new models in PyTorch XLA, see [Training PyTorch on Cloud TPUs](https://ultrons.medium.com/training-pytorch-on-cloud-tpus-be0649e4efbc) for a tutorial and [PyTorch on XLA Devices](https://pytorch.org/xla/release/1.10/index.html) for the APIs.

---

### Storage options

#### Filestore NFS

In the PyTorch XLA examples, we are using [Filestore](https://cloud.google.com/filestore) NFS for storage. One can create a Filestore NFS directory either from the [web console](https://console.cloud.google.com/filestore/instances?authuser=1) or from the [gcloud command line](https://cloud.google.com/filestore/docs/quickstart-gcloud). The NFS should be created in the same zone (e.g. `europe-west4-a` as your TPU VM is located) for efficient access.

Once an NFS directory is created, one can mount it on multiple TPU VM nodes to share code and data between them. For coordination between multiple users and projects, it is also recommended to create a dataset NFS to store common datasets (such as ImageNet) and share it between different users to avoid duplicated storage.

To mount an NFS directory (e.g. mounting `10.89.225.82:/mmf_megavlt` to `/checkpoint`):
```
sudo apt-get -y update
sudo apt-get -y install nfs-common
SHARED_FS=10.89.225.82:/mmf_megavlt
MOUNT_POINT=/checkpoint
sudo mkdir -p $MOUNT_POINT
sudo mount $SHARED_FS $MOUNT_POINT
sudo chmod go+rw $MOUNT_POINT
```

To unmount:
```
MOUNT_POINT=/checkpoint
sudo umount $MOUNT_POINT
```

#### Google Cloud Storage

An alternative storage option is [Google Cloud Storage](https://cloud.google.com/storage), where one can create a storage bucket [here](https://console.cloud.google.com/storage/browser?authuser=1) and copy data from and to the bucket using the gsutil tool following its [documentation](https://cloud.google.com/storage/docs/gsutil).

Storage buckets can be accessed from both your local laptops and the TPU VMs, or directly from the [web console](https://console.cloud.google.com/storage/browser?authuser=1), making it convenient to browse. One can use the storage buckets to transfer files between your laptop and TPU VMs (an alternative is to directly use scp, see "Useful commands for TPU VMs" below), or to backup data.

For example, to copy a file to Ronghang's bucket (`gs://ronghanghu_storage/checkpoint`):
```
gsutil cp ~/Downloads/xxx.zip gs://ronghanghu_storage/checkpoint
```
It is also useful to use `gsutil rsync` to backup or synchronize entire directories.

TensorFlow comes with built-in integration with Google Cloud Storage so that one can directly read from and write to the buckets in a TensorFlow training process. In PyTorch, it is harder to do so (and Filestore NFS will generally be a better option for PyTorch XLA training), but one can use the [google-cloud-storage Python API](https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python) if it is necessary.

#### Persistent disk

One can also create a [persistent disk](https://cloud.google.com/persistent-disk). When creating a standalone [compute engine VM instance](https://console.cloud.google.com/compute/instances?authuser=1), it will automatically create an associated persistent disk. Additional persistent disks can be created [here](https://console.cloud.google.com/compute/disks?authuser=1) and attached to a VM following the instructions [here](https://cloud.google.com/compute/docs/disks/add-persistent-disk).

Compared to Filestore NFS or Google Cloud Storage, it is often more difficult to share a persistent disk between multiple VM nodes (although it is possible in [some cases](https://cloud.google.com/compute/docs/disks/sharing-disks-between-vms)). Filestore NFS and Google Cloud Storage are usually more convenient for TPU VM use cases.

Filestore NFS, Google Cloud Storage, and persistent disk should be sufficient for most use cases. There are also other less common storage options. See [here](https://cloud.google.com/products/storage) for a full list.

---

### Useful commands for TPU VMs

#### Installing a dependency or running a command on all VM nodes

The startup script should have installed all the MAE dependencies. In case you want to install any new packages (e.g. pandas):
```
# (run on your TPU VM)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist --tpu=${TPU_NAME} -- \
  sudo pip3 install pandas
```

Or you can simultaneously SSH into all nodes in a pod and run the command there with `--worker all`, such as
```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --command "sudo pip3 install pandas" \
  --worker all
```

You can also manually SSH into a certain node in a pod with `--worker idx`. For example, to install pandas on worker 3:
```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --command "sudo pip3 install pandas" \
  --worker 3
```

#### Copying files from or to a TPU VM

One can use gcloud scp (see details [here](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/tpu-vm/scp)) to copy a file from or to a TPU VM. For example:
```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm scp --zone ${ZONE} --worker 0 \
   ~/Downloads/xxx.zip ${TPU_NAME}:/checkpoint/xxx.zip
```

#### Setting up VSCode connection

One can setup VSCode connection to the TPU VMs via [remote SSH connection](https://code.visualstudio.com/docs/remote/ssh).

First look up the external IP address of your TPM VM [here](https://console.cloud.google.com/compute/tpus?authuser=1) (e.g. `35.204.72.42`) and then use your `~/.ssh/google_compute_engine` key to set up a connection.
```
ssh 35.204.72.42 -i ~/.ssh/google_compute_engine
```

#### Deleting a TPU VM

To delete a TPU VM:
```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm delete ${TPU_NAME} --zone ${ZONE}
```

#### Using a remote coordinator VM to guard against maintenance events

For long-running jobs, one common reason for the job to fail is [maintenance events](https://cloud.google.com/tpu/docs/maintenance-events), which are beyond the users' control. When maintenance events happen, the connection to the TPU VM would often fail, and the training process will be lost.

To guard against maintenance events, one can also set up a remote coordinator (a standalone VM) following [prebuilt PyTorch XLA VM images](https://github.com/pytorch/xla#-consume-prebuilt-compute-vm-images) and [remote coordinator guide](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm#pods_with_remote_coordinator). When running `torch_xla.distributed.xla_dist` from the remote coordinator VM, it will check for TPU status and try to automatically recover from maintenance events by starting the training command again.

For this automatic resuming to be useful, **your training script needs to automatically search for the latest checkpoint and resume from it**. (In our MAE pretraining script above, we have `--resume automatic` for this purpose.)

On the remote coordinator VM, one can also mount the NFS storage. Then one can do VS code editing (or other CPU jobs like data zipping, etc) on the remote coordinator VM, and it is also possible to launch the TPU pod jobs from the remote coordinator (instead of directly launching them on the TPU VMs) via `torch_xla.distributed.xla_dist`, as in the links above.

#### Checking TPU status (busy/idle)

Sometimes when using more than one TPU pod, it could be easy to lose track of which TPU pods are busy and which ones are idle. A useful way to tell which TPU VMs are busy is to check for its current Python processes. Below is a script I use to check the status of all my TPU VMs.
```
# (run on your local laptop)

TPU_NAME_LIST="mae-tpu-128 mae-tpu-128-2"  # !!! change to the list of TPU VMs you're using
ZONE=europe-west4-a

PYTHON_PROCESS_NUM=8  # a busy node should have at least 8 python processes (for 8 TPU cores)

for TPU_NAME in $TPU_NAME_LIST; do
    out=$(gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker 0 --command "pgrep python | wc -l" 2>/dev/null)
    if [ -z $out ]; then
        STATUS="DOWN"
    elif [[ $out -ge $PYTHON_PROCESS_NUM ]]; then
        STATUS="BUSY"
    elif [[ $out -lt $PYTHON_PROCESS_NUM ]]; then
        STATUS="IDLE"
    else
        STATUS="UNKN"
    fi
    echo $STATUS $TPU_NAME
done
```
While this script doesn't cover all corner cases, I find it very reliable in practice.

One can also list all the TPUs in a zone as follows:
```
# (run on your local laptop)

ZONE=europe-west4-a
gcloud alpha compute tpus list --zone ${ZONE}
```
and query the IP address, health status, and other details of a VM as follows:
```
# (run on your local laptop)

TPU_NAME=mae-tpu-128  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm describe ${TPU_NAME} --zone ${ZONE}
```

#### Speed profiling and optimization

See [Cloud TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide) for general TPU performance information and [this doc](https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md) for specific PyTorch XLA information.

If your training is slower than expected, you can try using the TPU profiler to generate device traces and figure out the performance bottlenecks. See the guide for [PyTorch XLA](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm) (or [TensorFlow](https://cloud.google.com/tpu/docs/tensorflow-performance-guide)).
