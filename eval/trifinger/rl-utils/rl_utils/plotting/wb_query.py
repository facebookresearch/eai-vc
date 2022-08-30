try:
    import wandb
except ImportError:
    wandb = None
import os
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from rl_utils.common.core_utils import CacheHelper


def extract_query_key(k):
    if k.startswith("ALL_"):
        return k.split("ALL_")[1]
    return k


def query(
    select_fields: List[str],
    filter_fields: Dict[str, str],
    proj_cfg: Dict[str, Any],
    verbose=True,
    limit=None,
    use_cached=False,
    reduce_op: Optional[Callable[[List], float]] = None,
):
    """
    :param select_fields: The list of data to retrieve. If a field starts with
        "ALL_", then all the entries for this name from W&B are fetched. This gets
        the ENTIRE history.
    :param filter_fields: Key is the filter type (like group or tag) and value
        is the filter value (like the name of the group or tag to match)
    :param reduce_op: `np.mean` would take the average of the results.
    :param use_cached: Saves the results to disk so next time the same result is requested, it is loaded from disk rather than W&B.

    Selectable fields that can be in select_fields:
        - summary: The metrics for the model at the end of training. Also the
          run state. Useful if you want to check run result.
    """

    wb_proj_name = proj_cfg["proj_name"]
    wb_entity = proj_cfg["wb_entity"]

    lookup = f"{select_fields}_{filter_fields}"
    cache = CacheHelper("wb_queries", lookup)

    if use_cached and cache.exists():
        return cache.load()
    if wandb is None:
        raise ValueError("Wandb is not installed")

    api = wandb.Api()

    query_dict = {}
    search_id = None

    for f, v in filter_fields.items():
        if f == "group":
            query_dict["group"] = v
        elif f == "tag":
            query_dict["tags"] = v
        elif f == "id":
            search_id = v
        else:
            raise ValueError(f"Filter {f}: {v} not supported")

    def log(s):
        if verbose:
            print(s)

    if search_id is None:
        log("Querying with")
        log(query_dict)
        runs = api.runs(f"{wb_entity}/{wb_proj_name}", query_dict)
    else:
        log(f"Searching for ID {search_id}")
        runs = [api.run(f"{wb_entity}/{wb_proj_name}/{search_id}")]

    log(f"Returned {len(runs)} runs")

    ret_data = []
    for run in runs:
        dat = {}
        for f in select_fields:
            if f == "last_model":
                model_path = osp.join(run.config["logger"]["save_dir"], run.name)
                if not osp.exists(model_path):
                    raise ValueError(f"Could not locate model folder {model_path}")
                model_idxs = [
                    int(model_f.split("ckpt.")[1].split(".pth")[0])
                    for model_f in os.listdir(model_path)
                    if model_f.startswith("ckpt.")
                ]
                if len(model_idxs) == 0:
                    raise ValueError(f"No models found under {model_path}")
                max_idx = max(model_idxs)
                final_model_f = osp.join(model_path, f"ckpt.{max_idx}.pth")
                v = final_model_f
            elif f == "final_train_success":
                # Will by default get the most recent train success metric, if
                # none exists then will get the most recent eval success metric
                # (useful for methods that are eval only)
                succ_keys = [
                    k
                    for k in list(run.summary.keys())
                    if isinstance(k, str)
                    and "success" in k
                    and "std" not in k
                    and "max" not in k
                    and "min" not in k
                ]
                train_succ_keys = [
                    k for k in succ_keys if "eval_final" in k or "eval_train" in k
                ]
                if len(train_succ_keys) > 0:
                    use_k = train_succ_keys[0]
                elif len(succ_keys) > 0:
                    use_k = succ_keys[0]
                else:
                    print(
                        "Could not find success key from ",
                        run.summary.keys(),
                        "Possibly due to run failure. Run status",
                        run.state,
                    )
                    return None
                v = run.summary[use_k]
            elif f == "summary":
                v = dict(run.summary)
                v["status"] = str(run.state)
                # Filter out non-primitive values.
                v = {
                    k: k_v for k, k_v in v.items() if isinstance(k_v, (int, float, str))
                }

            elif f == "status":
                v = run.state
            elif f == "config":
                v = run.config
            elif f == "id":
                v = run.id
            elif f.startswith("config."):
                config_parts = f.split("config.")
                v = run.config[config_parts[1]]
            else:
                if f.startswith("ALL_"):
                    fetch_field = extract_query_key(f)
                    df = run.history(samples=15000)
                    if fetch_field not in df.columns:
                        raise ValueError(
                            f"Could not find {fetch_field} in {df.columns} for query {filter_fields}"
                        )
                    v = df[["_step", fetch_field]]
                else:
                    v = run.summary[f]
            dat[f] = v
        ret_data.append(dat)
        if limit is not None and len(ret_data) >= limit:
            break

    cache.save(ret_data)
    if reduce_op is not None:
        reduce_data = defaultdict(list)
        for p in ret_data:
            for k, v in p.items():
                reduce_data[k].append(v)
        ret_data = {k: reduce_op(v) for k, v in reduce_data.items()}

    log(f"Got data {ret_data}")
    return ret_data


def query_s(
    query_str: str, proj_cfg: DictConfig, verbose=True, use_cached: bool = False
):

    select_s, filter_s = query_str.split(" WHERE ")
    select_fields = select_s.replace(" ", "").split(",")

    parts = filter_s.split(" LIMIT ")
    filter_s = parts[0]

    limit = None
    if len(parts) > 1:
        limit = int(parts[1])

    filter_fields = filter_s.replace(" ", "").split(",")
    filter_fields = [s.split("=") for s in filter_fields]
    filter_fields = {k: v for k, v in filter_fields}

    return query(
        select_fields,
        filter_fields,
        proj_cfg,
        verbose=verbose,
        limit=limit,
        use_cached=use_cached,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--cache", action="store_true")
    args, query_args = parser.parse_known_args()
    query_args = " ".join(query_args)
    proj_cfg = OmegaConf.load(args.cfg)

    result = query_s(query_args, proj_cfg, use_cached=args.cache, verbose=False)
    pprint(result)
