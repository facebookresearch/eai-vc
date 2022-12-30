import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

import utils.data_utils as d_utils

from utils.encoder_model import (
    EncDecModel,
    EncoderModel,
)
import eaif_models

"""
Preprocess trajectories in demo directory with the structure:

demo_dir/
    difficulty_*/
        demo-*/
            demo-*.npz     # Raw observations from simulation (list of observation dicts)
            downsample.pth # Downsampled sim trajectories (dict of lists)
            <model>.pth    # Latent-state traj; downsample.pth trajectory passed through <model>

Example command to downsample demos to a desired downsample timestep (dts):
python utils/preprocess_trajs.py --top_demo_dir /private/home/clairelchen/projects/demos_green_cube/ --dts 0.4 -v

args:
    --top_demo_dir: path to demo_dir/
    --dts (float): downsample timestep, should be divisible by 0.004 (sim timestep)
    -v (optional flag): if specified, will save gifs of downsampled trajectories in each demo-*/
"""

SCALE = 100
DOWNSAMPLE_FILE_NAME = "downsample.pth"

EAIF_MODEL_NAMES = eaif_models.eaif_model_zoo

# Dict with custom_model_name: path_to_ckpt
# Model names must start with a prefix from CUSTOM_MODEL_PREFIXES
CUSTOM_MODEL_NAMES = {
    ## Trained on trifingerpro - colored_cube
    "encdec_1_pr-r3m_ld-16_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/49/ckpts/epoch_1000_ckpt.pth",
    "encdec_1_pr-r3m_ld-64_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/53/ckpts/epoch_1000_ckpt.pth",
    "encdec_1_pr-r3m_ld-256_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/57/ckpts/epoch_1000_ckpt.pth",
    "encdec_1_pr-r3m_ld-null_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/61/ckpts/epoch_1000_ckpt.pth",
    "encdec_1_pr-vip_ld-16_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/51/ckpts/epoch_1000_ckpt.pth",
    "encdec_1_pr-vip_ld-64_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/23/ckpts/epoch_1000_ckpt.pth",
    "encdec_1_pr-vip_ld-256_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/27/ckpts/epoch_1000_ckpt.pth",
    "encdec_1_pr-vip_ld-null_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-08/exp_enc_dec_1-2022-10-08-11-19-48/63/ckpts/epoch_1000_ckpt.pth",
    ## Trained on trifinger_meta - green_cube
    "encdec_2_pr-r3m_ld-16_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-25/exp_enc_dec_2-2022-10-25-20-05-25/49/ckpts/epoch_1000_ckpt.pth",
    "encdec_2_pr-r3m_ld-64_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-25/exp_enc_dec_2-2022-10-25-20-05-25/53/ckpts/epoch_1000_ckpt.pth",
    "encdec_2_pr-r3m_ld-256_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-25/exp_enc_dec_2-2022-10-25-20-05-25/57/ckpts/epoch_1000_ckpt.pth",
    "encdec_2_pr-r3m_ld-null_freeze-f": "/private/home/clairelchen/projects/eai-foundations/eval/trifinger/trifinger_claire/multirun/2022-10-25/exp_enc_dec_2-2022-10-25-20-05-25/29/ckpts/epoch_1000_ckpt.pth",
}

CUSTOM_MODEL_DECODERS = {}

CUSTOM_MODEL_PREFIXES = [
    "encdec",  # Load EncDecModel
    "bc",  # bc_finetune models
]

MODEL_NAMES = EAIF_MODEL_NAMES + list(CUSTOM_MODEL_NAMES.keys())


def get_demo_pth_list(top_demo_dir):
    """Get list of demo-*.npz file paths"""

    demo_pth_list = []
    # Find all demo-*.npz files
    for item_in_demo_dir in os.listdir(top_demo_dir):
        diff_dir = os.path.join(top_demo_dir, item_in_demo_dir)
        if os.path.isdir(diff_dir) and "difficulty" in item_in_demo_dir:
            for item_in_diff_dir in os.listdir(diff_dir):
                demo_dir = os.path.join(diff_dir, item_in_diff_dir)
                if os.path.isdir(demo_dir) and "demo" in item_in_diff_dir:
                    demo_name = f"{item_in_diff_dir}.npz"
                    demo_pth = os.path.join(demo_dir, demo_name)
                    if os.path.exists(demo_pth):
                        demo_pth_list.append(demo_pth)

    return demo_pth_list


def save_downsampled_traj_dict(demo_pth, dts):

    demo_dir = os.path.dirname(demo_pth)
    dts_dir = os.path.join(demo_dir, get_dts_dir_name(dts))
    if not os.path.exists(dts_dir):
        os.makedirs(dts_dir)

    file_pth = os.path.join(dts_dir, DOWNSAMPLE_FILE_NAME)

    if os.path.exists(file_pth):
        return

    # Generate and save downsample traj
    data = np.load(demo_pth, allow_pickle=True)["data"]
    traj_dict = d_utils.get_traj_dict_from_obs_list(
        data, scale=SCALE, include_image_obs=True
    )
    ds_traj_dict = d_utils.downsample_traj_dict(
        traj_dict,
        new_time_step=dts,
    )

    torch.save(ds_traj_dict, file_pth)
    # print(f"Saved downsampled traj to {file_pth}")


def save_rgb_gif(demo_pth, dts, cam_name="image_60"):

    demo_dir = os.path.dirname(demo_pth)
    dts_dir = os.path.join(demo_dir, get_dts_dir_name(dts))
    if not os.path.exists(dts_dir):
        os.makedirs(dts_dir)
    file_pth = os.path.join(dts_dir, DOWNSAMPLE_FILE_NAME)

    if not os.path.exists(file_pth):
        save_downsampled_traj_dict(demo_pth, dts)

    # Load downsampled traj dict
    traj_dict = torch.load(file_pth)

    gif_file_name = f"rgb_{cam_name}.gif"
    gif_file_pth = os.path.join(dts_dir, gif_file_name)

    if os.path.exists(gif_file_pth):
        return

    imgs = traj_dict["image_60"]
    d_utils.save_gif(imgs, gif_file_pth)


def save_encoded_imgs(demo_pth, model, transform, model_name, dts, cam_name="image_60"):

    file_name = f"{model_name}.pth"
    demo_dir = os.path.dirname(demo_pth)
    dts_dir = os.path.join(demo_dir, get_dts_dir_name(dts))
    if not os.path.exists(dts_dir):
        os.makedirs(dts_dir)
    file_pth = os.path.join(dts_dir, file_name)

    # Load downsampled traj dict
    traj_dict = torch.load(os.path.join(dts_dir, DOWNSAMPLE_FILE_NAME))

    if os.path.exists(file_pth):
        print(f"Found data in {file_pth}")
        data = torch.load(file_pth)["data"]
        print(len(data))
        print(data[0].shape)
        print(np.max(data[0]), np.min(data[0]))
        return

    data_list = []
    for img in traj_dict[cam_name]:
        encoded_img = d_utils.encode_img(model, transform, img)
        data_list.append(encoded_img.cpu().numpy())
        # print(torch.max(encoded_img), torch.min(encoded_img))
        # quit()

    torch.save({"data": np.array(data_list)}, file_pth)


def main(args):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Get list of demo-*.npz file paths
    demo_pth_list = get_demo_pth_list(args.top_demo_dir)

    # Create or make sure downsampled traj dict is saved for each demo
    print("Checking for or creating downsampled demos")
    for demo_pth in tqdm(demo_pth_list):
        save_downsampled_traj_dict(demo_pth, args.dts)

    if args.visualize:
        print("Generating gifs")
        for demo_pth in tqdm(demo_pth_list):
            save_rgb_gif(demo_pth, args.dts)

    for model_name in args.models:

        model, transform, _ = get_model_and_transform(model_name, device=device)
        model.eval()

        print(f"Checking for or creating {model_name} latent state demos")
        for demo_pth in tqdm(demo_pth_list):
            save_encoded_imgs(demo_pth, model, transform, model_name, args.dts)


def get_model_and_transform(model_name, device="cpu"):
    ## Pretrained EAIF models
    if model_name in EAIF_MODEL_NAMES:
        model, transform, latent_dim = d_utils.get_eaif_model_and_transform(
            model_name, device=device
        )

    ## Custom / finetuned models
    elif model_name in CUSTOM_MODEL_NAMES:

        custom_model_ckpt = CUSTOM_MODEL_NAMES[model_name]
        ckpt_info = torch.load(custom_model_ckpt)

        model_prefix = get_custom_model_prefix(model_name)

        # Load EncDecModel and ckpt weights
        if model_prefix == "encdec":
            conf = ckpt_info["conf"]

            pretrained_rep = conf.algo.pretrained_rep

            _, transform, embedding_dim = d_utils.get_eaif_model_and_transform(
                pretrained_rep, device=device
            )

            model_state_dict = ckpt_info["model_state_dict"]
            encdec_model = EncDecModel(
                pretrained_rep=pretrained_rep,
                freeze_pretrained_rep=conf.algo.freeze_pretrained_rep,
                latent_dim=conf.algo.latent_dim,
            )
            encdec_model.load_state_dict(model_state_dict)

            # Just take encoder
            model = encdec_model.encoder_model

            latent_dim = conf.algo.latent_dim

        elif model_prefix == "bc":

            conf = ckpt_info["conf"]

            _, transform, embedding_dim = d_utils.get_eaif_model_and_transform(
                conf.obj_state_type, device=device
            )
            model_state_dict = ckpt_info["encoder"]
            model = EncoderModel(
                pretrained_rep=conf.obj_state_type,
                freeze_pretrained_rep=conf.freeze_pretrained_rep,
                latent_dim=conf.latent_dim,
            )
            model.load_state_dict(model_state_dict)

            latent_dim = conf.latent_dim

        else:
            raise NameError(f"{model_prefix} is not in {CUSTOM_MODEL_PREFIXES}")

    else:
        raise NameError("Invalid model_name")

    return model, transform, latent_dim


def get_custom_model_conf(model_name):

    if model_name in CUSTOM_MODEL_NAMES:
        custom_model_ckpt = CUSTOM_MODEL_NAMES[model_name]
        ckpt_info = torch.load(custom_model_ckpt)
        conf = ckpt_info["conf"]
    else:
        raise NameError(f"{model_name} not in CUSTOM_MODEL_NAMES")

    return conf


def get_custom_model_prefix(model_name):
    model_prefix = model_name.split("_")[0]
    return model_prefix


def get_dts_dir_name(downsample_time_step):
    dts_str = str(downsample_time_step).replace(".", "p")
    dts_dir_name = "dts-" + dts_str
    return dts_dir_name


def parse_args():

    parser = argparse.ArgumentParser()

    # Required for specifying training and test trajectories
    parser.add_argument(
        "--top_demo_dir",
        default=f"/private/home/clairelchen/projects/demos/",
        help="Directory containing demos",
    )

    # Model
    parser.add_argument(
        "--models",
        "-m",
        nargs="*",
        default=[],
        choices=MODEL_NAMES,
        help="Mode to save",
    )
    parser.add_argument("--dts", type=float, default=0.2, help="Downsample time step")

    parser.add_argument("--visualize", "-v", action="store_true", help="Save RGB gifs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
