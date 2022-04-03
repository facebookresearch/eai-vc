import os
from argparse import Namespace

import wandb


def setup_wandb_output_dir(args: Namespace):
    if args.wandb_mode == "disabled":
        return
    # if wandb is enabled setup output directory
    assert args.output_dir is not None
    args.output_dir = os.path.join(args.output_dir, args.wandb_name)


def setup_wandb(args, project="tmae_training"):
    resume = None
    wandb_filename = os.path.join(args.output_dir, "wandb_id.txt")
    if os.path.exists(wandb_filename):
        # if file exists, then we are resuming from a previous eval
        with open(wandb_filename, "r") as file:
            wandb_id = file.read().rstrip("\n")
        resume = "must"
    else:
        wandb_id = wandb.util.generate_id()
        os.makedirs(os.path.dirname(wandb_filename), exist_ok=True)
        with open(wandb_filename, "w") as file:
            file.write(wandb_id)

    wandb.init(
        id=wandb_id,
        dir=args.output_dir,
        project=project,
        config=args,
        name=args.wandb_name,
        mode=args.wandb_mode,
        resume=resume,
    )
