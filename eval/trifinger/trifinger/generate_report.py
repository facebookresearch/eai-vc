import wandb
import copy


class ExpGroup:
    def __init__(self, run):
        conf = run.config
        c = copy.deepcopy(conf)
        if "seed" in c:
            c.pop("seed")
            c.pop("logger")

        self.conf = c
        self.seeds = set()

        self.total_reward = 0
        self.total_success_rate = 0
        self.total_ftip_dist = 0
        self.num_exp = 0
        self.step_val = 150000

        self.add_run(run)

    def equals(self, other_run):
        same_conf = False
        r = copy.deepcopy(other_run.config)
        if "seed" in r:
            r.pop("seed")
            r.pop("logger")
        if r == self.conf:
            same_conf = True
        return same_conf

    def add_run(self, new_run):
        seed = new_run.config["seed"]
        if seed in self.seeds:
            print(f"seed: {seed} already in run, skipping")
            return

        if self.step_val is None:
            hist = new_run.history()
            if hist.empty:
                return
            self.step_val = hist.iloc[-1]["_step"]

            res = new_run.scan_history(
                keys=["reward", "success_rate", "fingertip_dist", "_step"],
                page_size=1,
                min_step=self.step_val - 100,
                max_step=self.step_val + 100,
            )
            if res.max_step == 0:  # or run_data.next()['_step'] != self.step_val):
                print("Run did not have the correct step idx: ")
                return
            try:
                run_data = res.next()
            except StopIteration:
                print("Unable to add run! Skipping")
                return
        else:
            res = new_run.scan_history(
                keys=["reward", "success_rate", "fingertip_dist", "_step"],
                page_size=1,
                min_step=self.step_val - 100,
                max_step=self.step_val + 100,
            )
            # page_size=1)
            print("scan history returned")
            if res.max_step == 0:  # or run_data.next()['_step'] != self.step_val):
                print("Run did not have the correct step idx: ")
                return
            try:
                run_data = res.next()
            except StopIteration:
                print("Unable to add run! Skipping")
                return
        print(run_data)
        self.num_exp += 1
        self.total_reward += run_data["reward"]
        self.total_success_rate += run_data["success_rate"]
        self.total_ftip_dist += run_data["fingertip_dist"]
        self.seeds.add(seed)
        # print(self.get_stats())

    def get_stats(self):
        summary = {}
        # summary["conf"] = self.conf
        if "model" in self.conf:
            summary["model"] = self.conf["model"]["metadata"]
        summary["hidden_size"] = self.conf["policy"]["hidden_size"]
        summary["lr"] = self.conf["policy_updater"]["optimizer_params"]["lr"]
        summary["std_init"] = self.conf["policy"]["std_init"]
        summary["step_idx"] = self.step_val
        summary["num runs"] = self.num_exp
        if self.num_exp != 0:
            summary["avg_reward"] = self.total_reward / self.num_exp
            summary["avg_dist"] = self.total_success_rate / self.num_exp
            summary["avg_ftip_dist"] = self.total_ftip_dist / self.num_exp
        return summary


def matches_hyperparams(rconf):
    matches = True
    if rconf["policy_updater"]["optimizer_params"]["lr"] != 1e-05:
        matches = False
    if rconf["policy"]["std_init"] != -2:
        matches = False
    return matches


def main():
    api = wandb.Api(timeout=9)

    entity = "snsilwal"
    project = "10_25_trifinger_reach_VIP"
    # project = "10_24_trifinger_reach_new_action"  # project = "10_17_models_reach"
    # pass in a project name
    project_name = entity + "/" + project
    runs = api.runs(project_name)
    summary_list, config_list, name_list = [], [], []
    exp_groups = []
    # retrieve all of the runs associated with it
    for run in runs:
        single_run = run
        r_conf = run.config  # is a dict

        if not matches_hyperparams(r_conf):
            print("does not match")
            continue
        added_to_group = False
        for g in exp_groups:
            if g.equals(run):
                g.add_run(run)
                added_to_group = True

        if not added_to_group:
            exp_groups.append(ExpGroup(run))

    for g in exp_groups:
        print(g.get_stats())

    # group by config values
    # need to log average reward (training and testing), success rate, distance to goal


main()
