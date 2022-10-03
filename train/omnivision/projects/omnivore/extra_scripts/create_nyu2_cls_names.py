import os

import pickle


# Source for the class mapping for NYU (27 or 10 class version)
# https://github.com/zanilzanzan/FuseNet_PyTorch/blob/master/utils/text/nyu_scene_mapping.txt

raw_mapping = """
Class_Type              ID      Mapped_To   Sample_Amount
basement              1           10          7
bathroom              2           4           121
bedroom               3           1           383
bookstore             4           9           36
cafe                  5           10          5
classroom             6           8           49
computer_lab          7           10          6
conference_room       8           10          5
dinette               9           10          4
dining_room           10          5           117
exercise_room         11          10          3
foyer                 12          10          4
furniture_store       13          10          27
home_office           14          7           50
home_storage          15          10          5
indoor_balcony        16          10          2
kitchen               17          2           225
laundry_room          18          10          3
living_room           19          3           221
office                20          6           78
office_kitchen        21          10          10
playroom              22          10          31
printer_room          23          10          3
reception_room        24          10          17
student_lounge        25          10          5
study                 26          10          25
study_room            27          10          7
"""


def main():
    dt = raw_mapping.strip().split("\n")
    cls_mapped_id_to_name = {}
    for line in dt[1:]:
        clsname, orig_id, mapped_id, _ = line.strip().split()
        orig_id = int(orig_id)
        mapped_id = int(mapped_id)
        clsname = clsname.replace("_", " ")
        if mapped_id not in cls_mapped_id_to_name:
            cls_mapped_id_to_name[mapped_id] = []
        cls_mapped_id_to_name[mapped_id].append(clsname)

    # max number of names for a class
    max_num = 0
    for k in cls_mapped_id_to_name:
        max_num = max(len(cls_mapped_id_to_name[k]), max_num)

    cls_idxs = list(cls_mapped_id_to_name.keys())
    cls_idxs.sort()

    clsnames = []
    for k in cls_idxs:
        clsnames.append(cls_mapped_id_to_name[k])

    dst_dir = "/fsx-omnivore/imisra/datasets/nyuv2_cls"
    dst_file = os.path.join(dst_dir, "classnames_10cls.pkl")
    with open(dst_file, "wb") as fh:
        pickle.dump(clsnames, fh)


if __name__ == "__main__":
    main()
