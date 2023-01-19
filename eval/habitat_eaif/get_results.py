# launch runs in habitat_eaif

import os

if __name__ == "__main__":
    pre_path = "/checkpoint/yixinlin/eaif/results/run_habitat_eaif/karmeshyadav/"
    paths = [
        "clip_vit_base_patch16_wit_frozen_run_il",
        "mae_vit_base_patch16_['ego', 'imagenet', 'inav']_frozen_run_il",
        "mae_vit_large_patch16_['ego', 'imagenet', 'inav']_frozen_run_il",
        "mae_vit_base_patch16_['ego', 'inav']_frozen_run_il",
        "mae_vit_large_patch16_['ego', 'inav']_frozen_run_il",
        "mae_vit_base_patch16_['ego']_frozen_run_temp",
        "mae_vit_large_patch16_['ego']_frozen_run_il",
        "mae_vit_base_patch16_['inav']_frozen_run_il",
        "mae_vit_large_patch16_['inav']_frozen_run_il",
        "mvp_vit_base_patch16_['ego4d', 'hoi', 'imagenet']_frozen_run_il",
        "mvp_vit_large_patch16_['ego4d', 'hoi', 'imagenet']_frozen_run_il",
        "rand_vit_base_none_frozen_run_il",
        "rand_vit_large_none_frozen_run_il",
        "r3m_vit_base_patch16_ego4d_frozen_run_il",
    ]
    # paths = ["mae_vit_large_patch16_[\'ego\', \'imagenet\', \'inav\']_frozen_run_il"]
    # paths = ["mae_vit_large_patch16_[\'inav\']_frozen_run_il"]

    for path in paths:
        print("Path: {}".format(path))
        post_path = "/.submitit/"

        run_path = pre_path + path + post_path

        max_success = 0
        max_success_ckpt = 0
        max_ckpt = 0
        flag = 0
        # check all folders in the path
        for folder in os.listdir(run_path):
            # check if the folder is a directory
            if os.path.isdir(run_path + folder):
                # check if the folder contains a file named "stderr"
                log_file = run_path + folder + "/" + folder + "_0_log.err"
                if os.path.isfile(log_file):
                    # load the file
                    with open(log_file, "r") as f:
                        # find line which contains "Average episode success"
                        for line in f:
                            if "Checkpoint path" in line:
                                ckpt = line.split(" ")[-1].split("/")[-1].split(".")[1]
                                # print("Checkpoint Number: {}\n".format(ckpt))
                                if flag == 1:
                                    max_success_ckpt = ckpt
                                    flag = 0

                                if max_ckpt < int(ckpt):
                                    max_ckpt = int(ckpt)

                            if "Average episode success" in line:
                                success = line.split(" ")[-1][:-2]
                                # print("Success: {}\n".format(success))
                                if float(line.split(" ")[-1][:-2]) > max_success:
                                    max_success = float(line.split(" ")[-1][:-2])
                                    flag = 1

        print(
            "Max Success: {} | Ckpt: {} | Max Ckpt: {}\n\n".format(
                max_success, max_success_ckpt, max_ckpt
            )
        )
