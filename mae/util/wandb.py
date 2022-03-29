import os
from argparse import ArgumentParser, Namespace

import wandb


def setup_wandb_parser(parser: ArgumentParser):
    parser.add_argument(
        "--wandb_name",
        default="",
        type=str,
        help="name to be used for wandb logging",
    )
    parser.add_argument(
        "--wandb_mode",
        default="online",
        type=str,
        help="wandb mode to use for storing data,"
        "choose online, offline and disabled",
    )


def setup_wandb_args(args: Namespace):
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.wandb_name)


def setup_wandb(args, project="mae_training"):
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
        project=project,
        config=args,
        mode=args.wandb_mode,
        resume=resume,
    )
    wandb.run.name = args.wandb_name
    wandb.run.save()
