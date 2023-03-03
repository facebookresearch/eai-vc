#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import wandb
from tqdm import tqdm
import pandas as pd

pd.set_option("display.max_colwidth", None)


class WandbMetrics:
    def __init__(self, project, entity="cortexbench"):
        api = wandb.Api()
        self.runs = api.runs(entity + "/" + project)

    def extract_metrics(self, metric, filter_tags=[]):
        print(f"Extracting {metric} metrics")
        dfs = []
        for run in tqdm(self.runs):
            if any([tag in run.tags for tag in filter_tags]):
                continue

            steps, metrics = [], []
            for i, row in run.history(keys=[metric]).iterrows():
                steps.append(row["_step"])
                metrics.append(row[metric])

            dfs.append(
                pd.DataFrame(
                    data={"model_name": run.name, "step": steps, "metric": metrics}
                )
            )

        return pd.concat(dfs)

    def extract_data(self, train_metric, eval_metric, tags=[]):
        self.eval_metrics = self.extract_metrics(eval_metric, tags)
        self.train_metrics = self.extract_metrics(train_metric, tags)

    def clean_model_name(self, model_name):
        return " ".join(model_name.split("_")[0:-6])

    def get_metrics(self, max_step):
        # Compare all runs for a fixed step
        eval_metrics = self.eval_metrics.query(f"step < {max_step}")
        train_metrics = self.train_metrics.query(f"step < {max_step}")

        # Extract the max evaluation metric
        final_metrics = eval_metrics.groupby("model_name").max().reset_index()
        final_metrics["train_metric"] = 0
        final_metrics["max_train_step"] = 0
        final_metrics["max_test_step"] = 0

        # Get the closest train metric to the max eval metric
        for k, row in final_metrics.iterrows():
            run = row["model_name"]
            run_train_metrics = train_metrics[train_metrics["model_name"] == run]

            # Get closest train metric to max eval
            # train_metric = run_train_metrics.iloc[abs(run_train_metrics['step'] - row['step']).idxmin()]['metric']

            # Get max metric from train
            train_metric = run_train_metrics["metric"].max()

            final_metrics.loc[k, "train_metric"] = train_metric
            final_metrics.loc[k, "max_train_step"] = run_train_metrics["step"].max()
            final_metrics.loc[k, "max_test_step"] = eval_metrics[
                eval_metrics["model_name"] == run
            ]["step"].max()

        # Clean model name
        # final_metrics['model_name'] = final_metrics['model_name'].apply(self.clean_model_name)
        final_metrics["eval_metric"] = final_metrics["metric"]

        return (
            final_metrics[
                [
                    "model_name",
                    "train_metric",
                    "eval_metric",
                    "max_train_step",
                    "max_test_step",
                ]
            ]
            .sort_values("eval_metric", ascending=False)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # Project
    project = "habitat2.0"

    # Train and eval metrics
    train_metric, eval_metric = "metrics/pick_success", "eval_metrics/pick_success"

    # All runs with this task are old so shouln'd be reported
    filter_tags = ["Old"]

    # Extract data
    WB = WandbMetrics(project)
    WB.extract_data(train_metric, eval_metric, filter_tags)

    # I'm filtering all results after a fixed step horizon to do a fair comparison
    train_horizon = 500_000_000

    results = WB.get_metrics(train_horizon)
    results.to_csv(f"./results_{project}.csv", index=False)
