from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
from mjrl.policies.gaussian_mlp import MLP, BatchNormMLP
from mjrl.algos.behavior_cloning import BC
from eaif_mujoco.gym_wrapper import env_constructor
from eaif_mujoco.model_loading import (
    load_pretrained_model,
    fuse_embeddings_concat,
    fuse_embeddings_flare,
)
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from eaif_models.utils.wandb import setup_wandb
import mj_envs, gym, mjrl.envs, dmc2gym
import numpy as np, time as timer, multiprocessing, pickle, os, torch, gc
import torch.nn as nn
import torchvision.transforms as T


def set_seed(seed=None):
    """
    Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


def configure_cluster_GPUs(gpu_logical_id: int) -> int:
    """
    Maps the GPU logical ID to physical ID. This is required for MuJoCo to
    correctly use the GPUs, since it relies on physical ID unlike pytorch
    """
    # get the correct GPU ID
    if "SLURM_STEP_GPUS" in os.environ.keys():
        physical_gpu_ids = os.environ.get("SLURM_STEP_GPUS")
        gpu_id = int(physical_gpu_ids.split(",")[gpu_logical_id])
        print("Found slurm-GPUS: <Physical_id:{}>".format(physical_gpu_ids))
        print(
            "Using GPU <Physical_id:{}, Logical_id:{}>".format(gpu_id, gpu_logical_id)
        )
    else:
        gpu_id = 0  # base case when no GPUs detected in SLURM
        print("No GPUs detected. Defaulting to 0 as the device ID")
    return gpu_id


def bc_pvr_train_loop(config: dict) -> None:
    # configure GPUs
    # os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = configure_cluster_GPUs(config["env_kwargs"]["render_gpu_id"])
    config["env_kwargs"]["render_gpu_id"] = physical_gpu_id

    # set the seed
    set_seed(config["seed"])

    # infer the demo location
    demo_paths_loc = os.path.join(
        config["data_dir"], config["env_kwargs"]["env_name"] + ".pickle"
    )
    try:
        demo_paths = pickle.load(open(demo_paths_loc, "rb"))
    except:
        print("Unable to load the data. Check the data path.")
        print(demo_paths_loc)
        quit()

    demo_paths = demo_paths[: config["num_demos"]]
    demo_score = np.mean([np.sum(p["rewards"]) for p in demo_paths])
    print("Number of demonstrations used : %i" % len(demo_paths))
    print("Demonstration score : %.2f " % demo_score)

    # construct the environment and policy
    env_kwargs = config["env_kwargs"]
    e = env_constructor(**env_kwargs, fuse_embeddings=fuse_embeddings_flare)
    policy = BatchNormMLP(
        env_spec=e.spec,
        hidden_sizes=eval(config["bc_kwargs"]["hidden_sizes"]),
        seed=config["seed"],
        nonlinearity=config["bc_kwargs"]["nonlinearity"],
        dropout=config["bc_kwargs"]["dropout"],
    )

    # compute embeddings and create dataset
    print("===================================================================")
    print(">>>>>>>>> Precomputing frozen embedding dataset >>>>>>>>>>>>>>>>>>>")
    demo_paths = compute_embeddings(
        demo_paths,
        device=config["device"],
        embedding_name=config["env_kwargs"]["embedding_name"],
    )
    demo_paths = precompute_features(
        demo_paths,
        history_window=config["env_kwargs"]["history_window"],
        fuse_embeddings=fuse_embeddings_flare,
    )
    gc.collect()  # garbage collection to free up RAM
    dataset = FrozenEmbeddingDataset(
        demo_paths,
        history_window=config["env_kwargs"]["history_window"],
        fuse_embeddings=fuse_embeddings_flare,
    )
    # Dataset in this case is pre-loaded and on the RAM (CPU) and not on the disk
    dataloader = DataLoader(
        dataset,
        batch_size=config["bc_kwargs"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(
        list(policy.model.parameters()), lr=config["bc_kwargs"]["lr"]
    )
    loss_func = torch.nn.MSELoss()

    # Update logging to match eaif conventions
    # Make log dir
    wandb_run = setup_wandb(config)
    if os.path.isdir(config["job_name"]) == False:
        os.mkdir(config["job_name"])
    previous_dir = os.getcwd()
    os.chdir(config["job_name"])  # important! we are now in the directory to save data
    if os.path.isdir("iterations") == False:
        os.mkdir("iterations")
    if os.path.isdir("logs") == False:
        os.mkdir("logs")

    highest_score, highest_success = -np.inf, 0.0
    for epoch in tqdm(range(config["epochs"])):
        # move the policy to correct device
        policy.model.to(config["device"])
        policy.model.train()
        # update policy for one BC epoch
        running_loss = 0.0
        for mb_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            feat = batch["features"].float().to(config["device"])
            tar = batch["actions"].float().to(config["device"])
            pred = policy.model(feat)
            loss = loss_func(pred, tar.detach())
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.to("cpu").data.numpy().ravel()[0]
        # log average loss for the epoch
        wandb_run.log({"epoch_loss": running_loss / (mb_idx + 1)}, step=epoch + 1)
        # move the policy to CPU for saving and evaluation
        policy.model.to("cpu")
        policy.model.eval()
        # ensure enironment embedding is in eval mode before rollouts
        e.env.embedding.eval()

        # perform evaluation rollouts every few epochs
        if (epoch % config["eval_frequency"] == 0 and epoch > 0) or (
            epoch == config["epochs"] - 1
        ):
            paths = sample_paths(
                num_traj=config["eval_num_traj"],
                env=e,
                policy=policy,
                eval_mode=True,
                horizon=e.horizon,
                base_seed=config["seed"],
                num_cpu=config["num_cpu"],
            )
            mean_score = np.mean([np.sum(p["rewards"]) for p in paths])
            min_score = np.min([np.sum(p["rewards"]) for p in paths])
            max_score = np.max([np.sum(p["rewards"]) for p in paths])
            try:
                success_percentage = e.env.unwrapped.evaluate_success(paths)
            except:
                print("Success percentage function not implemented in env")
                success_percentage = -1

            highest_success = (
                success_percentage
                if success_percentage >= highest_success
                else highest_success
            )
            highest_score = mean_score if mean_score >= highest_score else highest_score
            epoch_log = {}
            epoch_log["eval_epoch"] = epoch
            epoch_log["eval_score_mean"] = mean_score
            epoch_log["eval_score_min"] = min_score
            epoch_log["eval_score_max"] = max_score
            epoch_log["eval_success"] = success_percentage
            epoch_log["highest_success"] = highest_success
            epoch_log["highest_score"] = highest_score
            wandb_run.log(data=epoch_log)

            print(
                "Epoch = %i | BC performance (eval mode) = %.3f " % (epoch, mean_score)
            )
            print(tabulate(sorted(epoch_log.items())))

        # save policy and logging
        if (epoch % config["save_frequency"] == 0 and epoch > 0) or (
            epoch == config["epochs"] - 1
        ):
            # pickle.dump(agent.policy, open('./iterations/policy_%i.pickle' % epoch, 'wb'))
            if highest_score == mean_score:
                pickle.dump(policy, open("./iterations/best_policy.pickle", "wb"))


class FrozenEmbeddingDataset(Dataset):
    def __init__(
        self,
        paths: list,
        history_window: int = 1,
        fuse_embeddings: callable = None,
        device: str = "cuda",
    ):
        self.paths = paths
        assert "embeddings" in self.paths[0].keys()
        # assume equal length trajectories
        # code will work even otherwise but may have some edge cases
        self.path_length = max([p["actions"].shape[0] for p in paths])
        self.num_paths = len(self.paths)
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        self.device = device

    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, self.paths[traj_idx]["actions"].shape[0])
        if "features" in self.paths[traj_idx].keys():
            features = self.paths[traj_idx]["features"][timestep]
            action = self.paths[traj_idx]["actions"][timestep]
        else:
            embeddings = [
                self.paths[traj_idx]["embeddings"][max(timestep - k, 0)]
                for k in range(self.history_window)
            ]
            embeddings = embeddings[
                ::-1
            ]  # embeddings[-1] should be most recent embedding
            features = self.fuse_embeddings(embeddings)
            # features = torch.from_numpy(features).float().to(self.device)
            action = self.paths[traj_idx]["actions"][timestep]
            # action   = torch.from_numpy(action).float().to(self.device)
        return {"features": features, "actions": action}


def compute_embeddings(
    paths: list, embedding_name: str, device: str = "cpu", chunk_size: int = 20
):
    model, embedding_dim, transforms, metadata = load_pretrained_model(
        embedding_name=embedding_name
    )
    model.to(device)
    for path in tqdm(paths):
        inp = path["images"]  # shape (B, H, W, 3)
        path["embeddings"] = np.zeros((inp.shape[0], embedding_dim))
        path_len = inp.shape[0]
        preprocessed_inp = torch.cat(
            [transforms(frame) for frame in inp]
        )  # shape (B, 3, H, W)
        for chunk in range(path_len // chunk_size + 1):
            if chunk_size * chunk < path_len:
                with torch.no_grad():
                    inp_chunk = preprocessed_inp[
                        chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                    ]
                    emb = model(inp_chunk.to(device))
                    # save embedding in RAM and free up GPU memory
                    emb = emb.to("cpu").data.numpy()
                path["embeddings"][
                    chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                ] = emb
        del path["images"]  # no longer need the images, free up RAM
    return paths


def precompute_features(
    paths: list,
    history_window: int = 1,
    fuse_embeddings: callable = None,
):
    assert "embeddings" in paths[0].keys()
    for path in paths:
        features = []
        for t in range(path["embeddings"].shape[0]):
            emb_hist_t = [
                path["embeddings"][max(t - k, 0)] for k in range(history_window)
            ]
            emb_hist_t = emb_hist_t[
                ::-1
            ]  # emb_hist_t[-1] should correspond to time t embedding
            feat_t = fuse_embeddings(emb_hist_t)
            features.append(feat_t.copy())
        path["features"] = np.array(features)
    return paths
