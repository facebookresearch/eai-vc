import os
import multiprocessing
import subprocess
import argparse
import time
from datetime import date, datetime

"""
Launch train.py jobs with various args
"""

def start_job(arg_dict):
    today_date = date.today().strftime("%m-%d-%y")
    timestamp = datetime.now().time().strftime("%H%M%S")

    filename = "{}_{}_{}".format(today_date, timestamp, arg_dict["i_run"])
    out_path = os.path.join("tmp/proc_logs", f"{filename}.out")
    err_path = os.path.join("tmp/proc_logs", f"{filename}.err")
        
    print("\n Process {}".format(arg_dict["i_run"]))
    print("Logging output to: {}".format(out_path))
    print("Logging errors to: {}".format(err_path))
    
    arg_str = arg_dict["arg_str"]
    cmd = f"python trifinger_mbirl/train.py {arg_str}"

    out_file = open(out_path, "w")
    err_file = open(err_path, "w")

    ret = subprocess.run(
         cmd, shell=True, stdout=out_file, stderr=err_file
     )
    return ret

def get_args_list(args):
    file_path_l = []
    # Iterate through .json split info files
    for item in os.listdir(args.file_path):
        if item.endswith(".json"):
            json_path = os.path.join(args.file_path, item)
            file_path_l.append(json_path)
     
    RUN_ID = 0
    arg_str_list = []

    ######################## BC args #########################
    #ALGO = "bc" 
    #bc_obs_type_l = ["goal_rel"]
    ##bc_obs_type_l = ["goal_rel", "img_r3m"]

    #for FILE_PATH in file_path_l:
    #    for BC_OBS_TYPE in bc_obs_type_l:
    #        arg_str = "\
    #                  --run_id {RUN_ID} \
    #                  --file_path {FILE_PATH} \
    #                  --no_wandb \
    #                  --algo {ALGO} \
    #                  --bc_obs_type {BC_OBS_TYPE} \
    #                  ".format(
    #                    RUN_ID       = RUN_ID, 
    #                    FILE_PATH    = FILE_PATH,
    #                    ALGO         = ALGO,
    #                    BC_OBS_TYPE  = BC_OBS_TYPE,
    #                    )
    #        arg_str_list.append({"i_run": RUN_ID, "arg_str": arg_str})
    #        RUN_ID += 1

    ######################## MBIRL args ########################
    ALGO = "mbirl" 
    MPC_TYPE = "two_phase"
    #cost_type_l = ["MPTimeDep"]
    cost_type_l = ["Traj", "Weighted", "TimeDep"]
    n_inner_iter_l = [1, 20, 50]
    cost_state_l = ["ftpos", "obj", "ftpos_obj"]
    irl_state_l = ["ftpos", "obj", "ftpos_obj"]
    EXP_ID = "two_phase_test_1" 

    for FILE_PATH in file_path_l:
        for COST_TYPE in cost_type_l:
            for COST_STATE in cost_state_l:
                for IRL_LOSS_STATE in irl_state_l:
                    for N_INNER_ITER in n_inner_iter_l:
                        arg_str = "\
                                  --exp_id {EXP_ID} \
                                  --run_id {RUN_ID} \
                                  --file_path {FILE_PATH} \
                                  --algo {ALGO} \
                                  --mpc_type {MPC_TYPE} \
                                  --cost_type {COST_TYPE} \
                                  --cost_state {COST_STATE} \
                                  --irl_loss_state {IRL_LOSS_STATE} \
                                  --n_inner_iter {N_INNER_ITER} \
                                  ".format(
                                    EXP_ID       = EXP_ID,
                                    RUN_ID       = RUN_ID, 
                                    FILE_PATH    = FILE_PATH,
                                    ALGO         = ALGO,
                                    MPC_TYPE     = MPC_TYPE,
                                    COST_TYPE    = COST_TYPE,
                                    COST_STATE   = COST_STATE,
                                    IRL_LOSS_STATE    = IRL_LOSS_STATE,
                                    N_INNER_ITER = N_INNER_ITER,
                                    )
                        arg_str_list.append({"i_run": RUN_ID, "arg_str": arg_str})
                        RUN_ID += 1

    return arg_str_list
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
    parser.add_argument('--n_jobs', type=int, default=1, help="""Number of jobs to launch""")
    return parser.parse_args()

def main(args):
    arg_str_list = get_args_list(args)

    with multiprocessing.Pool(processes=args.n_jobs) as pool:
        returns = pool.map(start_job, arg_str_list)

    process_return: subprocess.CompletedProcess

    error_count = 0
    for id, process_return in enumerate(returns):
        if process_return.returncode != 0:
            error_count += 1
            print(f"Failed with code {process_return.returncode}")
    
            #with open(os.path.join(error_log_dir, f"run_{id}.txt"), "w") as f:
            #  f.write(f"cmd >>> {process_return.args}\n")
            #  f.write(process_return.stdout.decode("utf-8"))
    
    print("All processes terminated")
    print(f"Total runs: {len(returns)} but {error_count} error(s)")

if __name__ == '__main__':
    args = parse_args()
    main(args)
