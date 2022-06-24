from matplotlib.pyplot import hist
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
from mjrl.policies.gaussian_mlp import MLP, BatchNormMLP
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.logger import DataLog
from visual_il.env_constructor import env_constructor
from visual_il.vision_model_loader import load_pvr_model, fuse_embeddings_concat, fuse_embeddings_flare
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import mj_envs, gym, mjrl.envs, dmc2gym
import numpy as np, time as timer, multiprocessing, pickle, os, torch, gc
import torch.nn as nn
import torchvision.transforms as T


def make_bc_agent(env_kwargs:dict, bc_kwargs:dict, demo_paths:list, epochs:int, seed:int):
    e = env_constructor(**env_kwargs)
    policy = MLP(e.spec, hidden_sizes=(256, 256), seed=seed)
    bc_agent = BC(demo_paths, policy=policy, epochs=epochs, set_transforms=False, **bc_kwargs)
    return e, bc_agent


def set_seed(seed=None):
    """
    Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def configure_cluster_GPUs(gpu_logical_id: int) -> int:
    """
    Maps the GPU logical ID to physical ID. This is required for MuJoCo to
    correctly use the GPUs, since it relies on physical ID unlike pytorch
    """
    # get the correct GPU ID
    if "SLURM_STEP_GPUS" in os.environ.keys():
        physical_gpu_ids = os.environ.get('SLURM_STEP_GPUS')
        gpu_id = int(physical_gpu_ids.split(',')[gpu_logical_id])
        print("Found slurm-GPUS: <Physical_id:{}>".format(physical_gpu_ids))
        print("Using GPU <Physical_id:{}, Logical_id:{}>".format(gpu_id, gpu_logical_id))
    else:
        gpu_id = 0 # base case when no GPUs detected in SLURM
        print("No GPUs detected. Defaulting to 0 as the device ID")
    return gpu_id


def bc_pvr_train_loop(job_data:dict) -> None:

    # configure GPUs
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = configure_cluster_GPUs(job_data['env_kwargs']['render_gpu_id'])
    job_data['env_kwargs']['render_gpu_id'] = physical_gpu_id

    # set the seed
    set_seed(job_data['seed'])

    # infer the demo location
    # the expert trajectories with pixel observations are assumed to be placed at
    # job_data['data_dir']/expert_paths/job_data['env_kwargs']['env_name'].pickle
    demo_paths_loc = job_data['data_dir'] + '/expert_paths/' + job_data['env_kwargs']['env_name'] + '.pickle'
    # demo_paths_loc = '/private/home/aravraj/work/Projects/visual_rl/vrl_private/vrl/hydra/expert_data/resnet18_rand_paths/relocate-v0.pickle'
    try:
        demo_paths = pickle.load(open(demo_paths_loc, 'rb'))
    except:
        print("Unable to load the data. Check the data path.")
        print(demo_paths_loc)
        quit()

    demo_paths = demo_paths[:job_data['num_demos']]
    demo_score = np.mean([np.sum(p['rewards']) for p in demo_paths])
    print("Number of demonstrations used : %i" % len(demo_paths))
    print("Demonstration score : %.2f " % demo_score)

    # construct the environment and policy
    env_kwargs = job_data['env_kwargs']
    e = env_constructor(**env_kwargs, fuse_embeddings=fuse_embeddings_flare)
    policy = BatchNormMLP(env_spec=e.spec, hidden_sizes=eval(job_data['bc_kwargs']['hidden_sizes']),
                          seed=job_data['seed'], nonlinearity=job_data['bc_kwargs']['nonlinearity'],
                          dropout=job_data['bc_kwargs']['dropout'])

    # compute embeddings and create dataset
    demo_paths = compute_embeddings(demo_paths, device=job_data['device'],
                                    embedding_name=job_data['env_kwargs']['embedding_name'])
    demo_paths = precompute_features(demo_paths, history_window=job_data['env_kwargs']['history_window'],
                                    fuse_embeddings=fuse_embeddings_flare)
    gc.collect()  # garbage collection to free up RAM
    dataset    = FrozenEmbeddingDataset(demo_paths,
                    history_window=job_data['env_kwargs']['history_window'],
                    fuse_embeddings=fuse_embeddings_flare,
                )
    dataloader = DataLoader(dataset, batch_size=job_data['bc_kwargs']['batch_size'], 
                            shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.Adam(list(policy.model.parameters()), lr=job_data['bc_kwargs']['lr'])
    loss_func = torch.nn.MSELoss()

    # Make log dir
    logger = DataLog()
    if os.path.isdir(job_data['job_name']) == False: os.mkdir(job_data['job_name'])
    previous_dir = os.getcwd()
    os.chdir(job_data['job_name']) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False: os.mkdir('logs')

    highest_score = -np.inf
    for epoch in tqdm(range(job_data['epochs'])):
        # move the policy to correct device
        policy.model.to(job_data['device'])
        policy.model.train()
        # update policy for one BC epoch
        running_loss = 0.0
        for mb_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            feat = batch['features'].float().to(job_data['device'])
            tar  = batch['actions'].float().to(job_data['device'])
            pred = policy.model(feat)
            loss = loss_func(pred, tar.detach())
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.to('cpu').data.numpy().ravel()[0]
        # log average loss for the epoch    
        logger.log_kv('epoch_loss', running_loss / (mb_idx+1))
        # move the policy to CPU for saving and evaluation
        policy.model.to('cpu')
        policy.model.eval()
        # ensure enironment embedding is in eval mode before rollouts
        e.env.embedding.eval()

        # perform evaluation rollouts every few epochs
        if (epoch % job_data['eval_frequency'] == 0 and epoch > 0) or (epoch == job_data['epochs']-1):
            paths = sample_paths(num_traj=job_data['eval_num_traj'], env=e, 
                                 policy=policy, eval_mode=True, horizon=e.horizon, 
                                 base_seed=job_data['seed'], num_cpu=job_data['num_cpu'])
            mean_score = np.mean([np.sum(p['rewards']) for p in paths])
            min_score  = np.min([np.sum(p['rewards']) for p in paths])
            max_score  = np.max([np.sum(p['rewards']) for p in paths])
            try:
                success_percentage = e.env.unwrapped.evaluate_success(paths)
            except:
                print("Success percentage function not implemented in env")
                success_percentage = -1
            logger.log_kv('eval_epoch', epoch)
            logger.log_kv('eval_score_mean', mean_score)
            logger.log_kv('eval_score_min', min_score)
            logger.log_kv('eval_score_max', max_score)
            logger.log_kv('eval_success', success_percentage)

            print("Epoch = %i | BC performance (eval mode) = %.3f " % (epoch, mean_score))

        # save policy and logging
        if (epoch % job_data['save_frequency'] == 0 and epoch > 0) or (epoch == job_data['epochs']-1):
            # pickle.dump(agent.policy, open('./iterations/policy_%i.pickle' % epoch, 'wb'))
            logger.save_log('./logs/')
            if mean_score > highest_score:
                pickle.dump(policy, open('./iterations/best_policy.pickle', 'wb'))
                highest_score = mean_score

            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                        logger.get_current_log().items()))
            print(tabulate(print_data))



class FrozenEmbeddingDataset(Dataset):
    def __init__(self, paths: list,
                 history_window: int = 1,
                 fuse_embeddings: callable = None,
                 device: str = 'cuda'):
        self.paths = paths
        assert 'embeddings' in self.paths[0].keys()
        # assume equal length trajectories
        # code will work even otherwise but may have some edge cases
        self.path_length = paths[0]['actions'].shape[0]
        self.num_paths = len(self.paths)
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        self.device = device
    
    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, self.paths[traj_idx]['actions'].shape[0])
        if 'features' in self.paths[traj_idx].keys():
            features = self.paths[traj_idx]['features'][timestep]
            action   = self.paths[traj_idx]['actions'][timestep]
        else:
            embeddings = [ self.paths[traj_idx]['embeddings'][max(timestep-k, 0)] for k in range(self.history_window) ]
            embeddings = embeddings[::-1]  # embeddings[-1] should be most recent embedding
            features = self.fuse_embeddings(embeddings)
            # features = torch.from_numpy(features).float().to(self.device)
            action   = self.paths[traj_idx]['actions'][timestep]
            # action   = torch.from_numpy(action).float().to(self.device)
        return {'features': features, 'actions': action}


def compute_embeddings(paths: list, embedding_name: str, 
                       device: str = 'cpu', chunk_size: int = 20):
    model, embedding_dim, transforms = load_pvr_model(embedding_name=embedding_name)
    model.to(device)
    for path in paths:
        inp = path['images']        # shape (B, H, W, 3)
        path['embeddings'] = np.zeros((inp.shape[0], embedding_dim))
        path_len = inp.shape[0]
        preprocessed_inp = torch.cat([transforms(frame) for frame in inp])   # shape (B, 3, H, W)
        for chunk in range(path_len // chunk_size + 1):
            if chunk_size * chunk < path_len:                
                with torch.no_grad():
                    inp_chunk = preprocessed_inp[chunk_size*chunk:min(chunk_size*(chunk+1),path_len)]
                    emb = model(inp_chunk.to(device))
                    emb = emb.to('cpu').data.numpy()      # shape (chunk_size, emb_dim)
                path['embeddings'][chunk_size*chunk:min(chunk_size*(chunk+1),path_len)] = emb
        del(path['images'])   # no longer need the images, free up RAM
    return paths


def precompute_features(paths: list,
                 history_window: int = 1,
                 fuse_embeddings: callable = None,
                ):
    assert 'embeddings' in paths[0].keys()
    for path in paths:
        features = []
        for t in range(path['embeddings'].shape[0]):
            emb_hist_t = [ path['embeddings'][max(t-k, 0)] for k in range(history_window) ]
            emb_hist_t = emb_hist_t[::-1]  # emb_hist_t[-1] should correspond to time t embedding
            feat_t = fuse_embeddings(emb_hist_t)
            features.append(feat_t.copy())
        path['features'] = np.array(features)
    return paths
